"""Rebuild marker assets from the company-provided PDF (pypdf required)."""
from pathlib import Path
import argparse
import io
import json
from PIL import Image
from pypdf import PdfReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', required=True)
    parser.add_argument('--out', default='assets')
    args = parser.parse_args()
    pdf = Path(args.pdf)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    images = list(PdfReader(pdf).pages[0].images)
    specs = [('marker_printed.png', 0, [113,74,150,98]),
             ('marker_dark.png', 1, [87,95,111,113]),
             ('marker_light.png', 2, [43,105,77,124])]
    manifest = dict(source='AI Evaluation Test.pdf, page 1, company-provided reference illustrations', templates=[])
    for name, index, box in specs:
        Image.open(io.BytesIO(images[index].data)).crop(box).save(out/name)
        manifest['templates'].append(dict(file=name, pdf_image_index=index, crop_xyxy=box))
    (out/'marker_references.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
