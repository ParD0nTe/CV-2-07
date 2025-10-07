"""
Поиск объекта по шаблону (template matching).

Шаги по заданию:
1) Берём исходное изображение (по умолчанию — skimage.data.astronaut()).
2) Вырезаем часть как шаблон.
3) Используем cv2.matchTemplate(..., method=cv2.TM_CCOEFF_NORMED).
4) Находим максимальное значение отклика и его координаты.
5) Рисуем прямоугольник на исходном изображении.

Аргументы CLI позволяют:
- Задать свой путь к изображению (--image_path).
- Указать прямоугольник шаблона (--tpl_rect x y w h).
- Задать метод сопоставления (--method).
- Сохранить результаты (--out1, --out2) и отключить показ окон (--no-show).
"""

import argparse
import sys
from typing import Tuple, Optional
import matplotlib.pyplot as plt

import numpy as np
import cv2
import matplotlib.pyplot as plt

try:
    from skimage import data as skdata
    HAVE_SKIMAGE = True
except Exception:
    HAVE_SKIMAGE = False


# ----------------------------- Утилиты -----------------------------

def cv2_method_from_str(name: str) -> int:
    """
    Преобразует строковое имя метода в константу OpenCV.
    Поддерживаются: TM_CCOEFF_NORMED, TM_CCOEFF, TM_CCORR_NORMED, TM_CCORR, TM_SQDIFF_NORMED, TM_SQDIFF.
    """
    name = name.strip().upper()
    mapping = {
        "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
        "TM_CCOEFF": cv2.TM_CCOEFF,
        "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
        "TM_CCORR": cv2.TM_CCORR,
        "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED,
        "TM_SQDIFF": cv2.TM_SQDIFF,
    }
    if name not in mapping:
        raise ValueError(f"Неизвестный метод '{name}'. Допустимо: {', '.join(mapping.keys())}")
    return mapping[name]

def select_roi_rect(img_rgb):
    """Интерактивный выбор ROI. Возвращает (x, y, w, h)."""
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    r = cv2.selectROI("Select template (press ENTER to confirm)", bgr, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select template (press ENTER to confirm)")
    x, y, w, h = map(int, r)
    if w == 0 or h == 0:
        raise ValueError("ROI не выбран (w=0 или h=0).")
    return x, y, w, h


def load_image_rgb(image_path: Optional[str]) -> np.ndarray:
    """
    Загружает изображение в RGB.
    Если путь не указан, пытается взять skimage.data.astronaut().
    """
    if image_path:
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение по пути: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb

    if HAVE_SKIMAGE:
        return skdata.astronaut()  # 512x512 RGB
    raise RuntimeError("Не передан --image_path и недоступен skimage.data.astronaut(). Установите scikit-image или укажите путь к файлу.")


def to_gray(img_rgb: np.ndarray) -> np.ndarray:
    """Преобразует RGB -> grayscale (uint8)."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return gray


def extract_template(img_rgb: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Вырезает шаблон из RGB-изображения по прямоугольнику (x, y, w, h).
    Возвращает шаблон в GRAY.
    """
    h, w = img_rgb.shape[:2]
    x, y, rw, rh = rect
    if rw <= 0 or rh <= 0:
        raise ValueError("Ширина и высота шаблона (w, h) должны быть > 0.")
    if x < 0 or y < 0 or x + rw > w or y + rh > h:
        raise ValueError(f"Шаблон выходит за границы изображения: img=({w}x{h}), rect={rect}")
    roi_rgb = img_rgb[y:y+rh, x:x+rw]
    return to_gray(roi_rgb)


def draw_rect(ax, x: int, y: int, w: int, h: int, color: str = "lime", lw: int = 2):
    """Рисует прямоугольник на matplotlib Axes."""
    import matplotlib.patches as patches
    rect = patches.Rectangle((x, y), w, h, linewidth=lw, edgecolor=color, facecolor="none")
    ax.add_patch(rect)


def add_noise(img_rgb: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    """
    Добавляет гауссов шум к RGB изображению (ради «другого» изображения).
    sigma ~ 5-15 обычно не ломает matching.
    """
    noise = np.random.normal(0, sigma, img_rgb.shape).astype(np.float32)
    out = np.clip(img_rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


# -------------------------- Основная логика -------------------------

def match_and_draw(
    img_rgb: np.ndarray,
    tpl_gray: np.ndarray,
    method_name: str = "TM_CCOEFF_NORMED",
    out_path: Optional[str] = None,
    title_suffix: str = "",
    show: bool = True,
) -> Tuple[np.ndarray, Tuple[int, int], float]:
    """
    Выполняет сопоставление шаблона и рисует прямоугольник на RGB-изображении.

    Возвращает:
      (annotated_img_rgb, (x, y), score)
      где (x, y) — координаты левого верхнего угла найденного шаблона,
      score — максимальный отклик matchTemplate (или минимальный для SQDIFF-методов).
    """
    method = cv2_method_from_str(method_name)

    # Преобразуем исходник в GRAY для matchTemplate
    img_gray = to_gray(img_rgb)
    res = cv2.matchTemplate(img_gray, tpl_gray, method)

    # Поиск лучшей позиции: для SQDIFF — минимальное значение, иначе — максимальное.
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    use_min = method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED)
    best_val = min_val if use_min else max_val
    best_loc = min_loc if use_min else max_loc

    th, tw = tpl_gray.shape[:2]
    x, y = best_loc

    # Рисуем прямоугольник на копии RGB
    annotated = img_rgb.copy()
    cv2.rectangle(annotated, (x, y), (x + tw, y + th), (0, 255, 0), thickness=2)

    # Визуализация
    if show:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].imshow(img_rgb)
        ax[0].set_title(f"Исходное изображение{title_suffix}")
        ax[0].axis("off")
        draw_rect(ax[0], x, y, tw, th, color="lime")

        ax[1].imshow(res, cmap="viridis")
        ax[1].set_title(f"Карта отклика ({method_name})\nscore={best_val:.3f}, loc={best_loc}")
        ax[1].axis("off")
        plt.tight_layout()
        plt.show()

    # Сохранение результата
    if out_path:
        bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(out_path, bgr)
        if not ok:
            print(f"[Предупреждение] Не удалось сохранить {out_path}", file=sys.stderr)

    return annotated, best_loc, float(best_val)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Template Matching: поиск объекта по шаблону (OpenCV)."
    )
    p.add_argument("--image_path", type=str, default=None,
                   help="Путь к исходному изображению. Если не задан, используется skimage.data.astronaut().")
    # Прямоугольник шаблона: x y w h. Для astronaut (512x512) глаз примерно так:
    p.add_argument("--tpl_rect", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                   default=[330, 200, 40, 25],
                   help="Прямоугольник шаблона: X Y W H (по умолчанию: 330 200 40 25 — глаз у astronaut)")
    p.add_argument("--method", type=str, default="TM_CCOEFF_NORMED",
                   choices=["TM_CCOEFF_NORMED", "TM_CCOEFF", "TM_CCORR_NORMED", "TM_CCORR", "TM_SQDIFF_NORMED", "TM_SQDIFF"],
                   help="Метод сопоставления (по умолчанию TM_CCOEFF_NORMED).")
    p.add_argument("--out1", type=str, default="match_result_1.png",
                   help="Путь для сохранения результата №1 (по умолчанию match_result_1.png).")
    p.add_argument("--out2", type=str, default="match_result_2.png",
                   help="Путь для сохранения результата №2 (по умолчанию match_result_2.png).")
    p.add_argument("--no-show", action="store_true",
                   help="Не показывать графики (только сохранить результаты).")
    p.add_argument("--noise-sigma", type=float, default=8.0,
                   help="Сигма шума для второго изображения (по умолчанию 8.0).")
    p.add_argument("--interactive", action="store_true",
               help="Выбрать шаблон мышкой (cv2.selectROI). Игнорирует --tpl_rect.")

    p.add_argument("--show-template", action="store_true",
               help="Показать окно с вырезанным шаблоном.")
    return p.parse_args()


def main() -> None:
    """
    Основной сценарий:
      1) Загружаем изображение (или берём astronaut).
      2) Вырезаем шаблон по прямоугольнику.
      3) Ищем шаблон на исходном изображении, печатаем координаты и score.
      4) Результаты показываем и/или сохраняем.
    """
    args = parse_args()

    try:
        # 1) Исходное изображение
        img_rgb = load_image_rgb(args.image_path)

        # 2) Вырезаем шаблон
        if args.interactive:
            x, y, w, h = select_roi_rect(img_rgb)
        else:
            x, y, w, h = args.tpl_rect
        tpl_gray = extract_template(img_rgb, (x, y, w, h))

        plt.figure(figsize=(3,3)); plt.imshow(tpl_gray, cmap="gray")
        plt.title(f"Template {w}x{h} at ({x},{y})"); plt.axis("off"); plt.show()

        if args.show_template:
            plt.figure(figsize=(3, 3))
            plt.imshow(tpl_gray, cmap="gray")
            plt.title(f"Template {tpl_gray.shape[1]}x{tpl_gray.shape[0]}")
            plt.axis("off")
            plt.show()

        # 3) Матчинг на исходнике
        ann1, loc1, score1 = match_and_draw(
            img_rgb=img_rgb,
            tpl_gray=tpl_gray,
            method_name=args.method,
            out_path=args.out1,
            title_suffix=" (оригинал)",
            show=not args.no_show,
        )
        print(f"[1] {args.method}: score={score1:.3f}, top-left={loc1}")

        # 4) Второе изображение — «другое» (добавим шум, чтобы не ломать масштаб)
        img2_rgb = add_noise(img_rgb, sigma=args.noise_sigma)
        ann2, loc2, score2 = match_and_draw(
            img_rgb=img2_rgb,
            tpl_gray=tpl_gray,
            method_name=args.method,
            out_path=args.out2,
            title_suffix=f" (шум σ={args.noise_sigma})",
            show=not args.no_show,
        )
        print(f"[2] {args.method} (noisy): score={score2:.3f}, top-left={loc2}")

        # Базовая метрика по заданию: координаты должны совпадать визуально.
        # Вблизи шума координаты могут слегка «гулять» — это нормально.

    except Exception as e:
        print(f"[Ошибка] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
