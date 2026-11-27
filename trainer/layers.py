"""
Расстановка (размещение) слоев — пиксельная сетка 16x16 из предсказаний модели.

Подход:
1. Модель возвращает цвета в "плоском" порядке row-major: пиксель i -> (y=i//16, x=i%16).
   Здесь этот порядок восстанавливается явно, чтобы расстановку можно было менять.
2. Каждый пиксель получает роль по эврике "фон/объект":
     - фон (background): цвет близок к среднему цвету канвы и низкая насыщенность;
     - объект (sprite): всё остальное.
3. Компонентный анализ (BFS) на маске объекта:
     - самая большая связная компонента = тело спрайта (core);
     - мелкие компоненты = мусор (noise), их убираем;
     - одиночные пиксели-одиночки вокруг тела = акценты (accent), оставляем только
       если они контрастны телу.
4. Сглаживание (cleanup):
     - у тела убираем "дырки" (fon внутри спрайта закрашиваем медианой тела);
     - срезается тонкий "бахрома" — пиксели тела, у которых <=1 соседа по телу.
5. Центрирование bounding box спрайта на канве (центр масс тяжёлых пикселей).

Функции чистые: получают numpy-массив [N, 3] цветов и возвращают ту же форму,
поэтому их можно вставлять в pipeline как есть и обучать модель уже "под них".
"""

import numpy as np

CANVAS = 16
NPIX = CANVAS * CANVAS


def flat_to_grid(pixels: np.ndarray) -> np.ndarray:
    """[256, 3] -> [16, 16, 3], row-major: i -> (i // 16, i % 16)."""
    return pixels.reshape(CANVAS, CANVAS, 3)


def grid_to_flat(grid: np.ndarray) -> np.ndarray:
    """[16, 16, 3] -> [256, 3]."""
    return grid.reshape(NPIX, 3)


# ---------------------------------------------------------------------------
# 1. Роли пикселей
# ---------------------------------------------------------------------------

def classify_roles(grid: np.ndarray):
    """Возвращает mask_obj (bool [16,16]) и bg_color."""
    # Фон = медианный цвет канвы (пиксели рядом с медианой считаются фоном)
    flat = grid.reshape(-1, 3)
    bg = np.median(flat, axis=0)
    dist = np.linalg.norm(flat - bg, axis=1)
    thr = max(0.15, np.percentile(dist, 60))  # не даём маске быть пустой/полной
    mask_obj = (dist > thr).reshape(CANVAS, CANVAS)
    return mask_obj, bg


# ---------------------------------------------------------------------------
# 2. Компонентный анализ
# ---------------------------------------------------------------------------

def _components(mask: np.ndarray):
    """Связные компоненты (4-связность) булевой маски -> список списков координат."""
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for y in range(CANVAS):
        for x in range(CANVAS):
            if not mask[y, x] or seen[y, x]:
                continue
            stack, comp = [(y, x)], []
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                comp.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < CANVAS and 0 <= nx < CANVAS and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append(comp)
    return comps


def split_components(mask_obj: np.ndarray):
    """core (самая большая компонента), accents (одиночки), noise (всё лишнее)."""
    comps = _components(mask_obj)
    if not comps:
        return np.zeros_like(mask_obj), np.zeros_like(mask_obj)
    biggest = max(comps, key=len)
    core_mask = np.zeros_like(mask_obj)
    for y, x in biggest:
        core_mask[y, x] = True
    accent_mask = np.zeros_like(mask_obj)
    for comp in comps:
        if comp is biggest:
            continue
        if len(comp) == 1:  # одиночный пиксель — кандидат в акценты
            y, x = comp[0]
            accent_mask[y, x] = True
        # компоненты размером 2..N без связи с телом — шум, удаляем
    return core_mask, accent_mask


# ---------------------------------------------------------------------------
# 3. Сглаживание
# ---------------------------------------------------------------------------

def fill_holes(core_mask: np.ndarray) -> np.ndarray:
    """Закрашивает фоновые 'дырки', полностью окружённые телом."""
    m = core_mask.copy()
    changed = True
    while changed:
        changed = False
        for y in range(1, CANVAS - 1):
            for x in range(1, CANVAS - 1):
                if not m[y, x]:
                    nb = [m[y - 1, x], m[y + 1, x], m[y, x - 1], m[y, x + 1]]
                    if sum(nb) >= 3:  # дырка, закрываем
                        m[y, x] = True
                        changed = True
    return m


def trim_fringe(core_mask: np.ndarray) -> np.ndarray:
    """Убирает пиксели тела с <=1 соседом по телу (торчащие 'ворсинки')."""
    m = core_mask.copy()
    for y, x in zip(*np.nonzero(core_mask)):
        n = sum(core_mask[ny, nx] for ny, nx in
                ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))
                if 0 <= ny < CANVAS and 0 <= nx < CANVAS)
        if n <= 1:
            m[y, x] = False
    return m


# ---------------------------------------------------------------------------
# 4. Центрирование
# ---------------------------------------------------------------------------

def center_sprite(grid: np.ndarray, core_mask: np.ndarray, accent_mask: np.ndarray,
                  bg_color: np.ndarray):
    """Сдвигает bbox спрайта в центр канвы целым сдвигом (без интерполяции)."""
    ys, xs = np.nonzero(core_mask | accent_mask)
    if len(ys) == 0:
        return grid
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    # центр масс вместо центра bbox — устойчиво к асимметричным спрайтам
    cy = int(round(ys.mean()))
    cx = int(round(xs.mean()))
    dy = CANVAS // 2 - cy
    dx = CANVAS // 2 - cx
    out = np.zeros_like(grid)
    out[:, :] = bg_color
    for y, x in zip(*np.nonzero(core_mask | accent_mask)):
        ny, nx = y + dy, x + dx
        if 0 <= ny < CANVAS and 0 <= nx < CANVAS:
            out[ny, nx] = grid[y, x]
    return out


# ---------------------------------------------------------------------------
# Главная функция расстановки
# ---------------------------------------------------------------------------

def arrange_layers(pixels: np.ndarray) -> np.ndarray:
    """Полный проход: [256,3] -> [256,3] после расстановки слоев.

    colorspace: RGB float в [0,1]. Прозрачность кодируется чёрным (0,0,0),
    как в dataset.py — это решение инференсера (alpha=0 при черном).
    """
    grid = flat_to_grid(np.asarray(pixels, dtype=np.float32))
    mask_obj, bg = classify_roles(grid)
    core_mask, accent_mask = split_components(mask_obj)
    core_mask = trim_fringe(fill_holes(core_mask))
    grid = center_sprite(grid, core_mask, accent_mask, bg)
    return grid_to_flat(grid)
