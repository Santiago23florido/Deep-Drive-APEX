#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def rounded_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    stroke_fill: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    x, y = xy
    bbox = draw.multiline_textbbox((x, y), text, font=font, spacing=6)
    pad_x = 18
    pad_y = 14
    rect = (
        bbox[0] - pad_x,
        bbox[1] - pad_y,
        bbox[2] + pad_x,
        bbox[3] + pad_y,
    )
    draw.rounded_rectangle(rect, radius=18, fill=fill, outline=stroke_fill, width=4)
    draw.multiline_text((x, y), text, font=font, fill=(255, 255, 255), spacing=6)
    return rect


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: tuple[int, int, int],
    width: int = 10,
) -> None:
    draw.line([start, end], fill=color, width=width)
    ex, ey = end
    sx, sy = start
    dx = ex - sx
    dy = ey - sy
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux = dx / length
    uy = dy / length
    left = (ex - int(24 * ux - 14 * uy), ey - int(24 * uy + 14 * ux))
    right = (ex - int(24 * ux + 14 * uy), ey - int(24 * uy - 14 * ux))
    draw.polygon([end, left, right], fill=color)


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate the trajectory image with the 09d failure mode.")
    parser.add_argument("input_image", type=Path)
    parser.add_argument("output_image", type=Path)
    args = parser.parse_args()

    image = Image.open(args.input_image).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")

    title_font = load_font(44)
    text_font = load_font(34)
    small_font = load_font(28)

    draw.rounded_rectangle((30, 30, 1110, 180), radius=24, fill=(20, 20, 20, 210))
    draw.text((60, 50), "Por que la trayectoria verde se abre demasiado", font=title_font, fill=(255, 255, 255))
    draw.text(
        (60, 110),
        "09d: entra bien en la curva, pero luego cambia de lectura y termina siguiendo el espacio abierto.",
        font=small_font,
        fill=(235, 235, 235),
    )

    rect1 = rounded_label(
        draw,
        (70, 320),
        "1. Entrada correcta\nfront_turn manda izquierda\n(target ~ +27 deg)",
        font=text_font,
        fill=(38, 92, 54),
        stroke_fill=(93, 214, 119),
    )
    arrow(draw, (rect1[2], rect1[1] + 85), (420, 520), color=(93, 214, 119))

    rect2 = rounded_label(
        draw,
        (80, 820),
        "2. Cambio de lectura\nfront_left cae y front_right sube\nel target cambia de signo",
        font=text_font,
        fill=(120, 70, 20),
        stroke_fill=(255, 178, 64),
    )
    arrow(draw, (rect2[2], rect2[1] + 100), (620, 760), color=(255, 178, 64))

    rect3 = rounded_label(
        draw,
        (660, 1120),
        "3. Gap takeover\nla logica sigue la apertura exterior\npor eso no sigue la roja",
        font=text_font,
        fill=(120, 30, 30),
        stroke_fill=(255, 90, 90),
    )
    arrow(draw, (rect3[0] + 120, rect3[1]), (925, 1290), color=(255, 90, 90))

    draw.rounded_rectangle((40, 1440, 1160, 1575), radius=20, fill=(15, 15, 15, 210))
    draw.text(
        (70, 1470),
        "Resumen: la version probada en 09d no estaba siguiendo un eje de corredor.\n"
        "Estaba haciendo front_turn -> cambio de signo -> gap.",
        font=small_font,
        fill=(255, 255, 255),
        spacing=8,
    )

    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output_image, quality=95)
    print(args.output_image)


if __name__ == "__main__":
    main()
