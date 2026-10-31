"""Contact-sheet helper: a labelled grid of rendered PNGs (for human/AI visual review)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def contact_sheet(items: list[tuple[Path | str, str]], out: Path | str, *, cols: int = 4, cell: tuple[int, int] = (360, 300),
                  label_h: int = 34, bg=(235, 235, 235)) -> Path:
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (cell[0] + 8) + 8, rows * (cell[1] + label_h + 8) + 8), bg)
    draw = ImageDraw.Draw(sheet)
    for i, (path, caption) in enumerate(items):
        col, row = i % cols, i // cols
        x, y = 8 + col * (cell[0] + 8), 8 + row * (cell[1] + label_h + 8)
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail(cell, Image.LANCZOS)
                canvas = Image.new("RGB", cell, (255, 255, 255))
                canvas.paste(im, ((cell[0] - im.width) // 2, (cell[1] - im.height) // 2))
                sheet.paste(canvas, (x, y + label_h))
                draw.rectangle([x, y + label_h, x + cell[0] - 1, y + label_h + cell[1] - 1], outline=(150, 150, 150))
        except OSError:
            draw.text((x + 4, y + label_h + 4), "missing preview", fill=(200, 0, 0))
        draw.text((x + 2, y + 2), caption[:60], fill=(0, 0, 0))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88) if out.suffix.lower() in (".jpg", ".jpeg") else sheet.save(out)
    return out
