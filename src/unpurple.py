import cv2
import numpy as np
import argparse


def unpurple(params):
    img = cv2.imread(params.input).astype(np.float64)

    img_b = img[..., 0]

    img_b = np.maximum(0, img_b - params.m * 255)
    img_b *= params.i / (1 - params.m)

    width = (params.r << 1) + 1

    bl = cv2.blur(img_b, (width, width))

    if params.mode == "blur":
        cv2.imwrite(params.output, bl)
    else:
        db = np.maximum(img[..., 0] - img[..., 1], 0)
        dr = np.maximum(img[..., 2] - img[..., 1], 0)

        mb = np.minimum(bl, db)

        r_diff = np.minimum(dr, mb * params.maxred)

        if params.minred > 0:
            b_diff = np.minimum(mb, r_diff / params.minred)
        else:
            b_diff = mb

        if params.mode == "diff":
            img_diff = np.dstack(
                [b_diff,
                 np.zeros_like(b_diff),
                 r_diff]
            )
            img_diff = img_diff.astype(np.uint8)
            cv2.imwrite(params.output, img_diff)
        else:
            assert(params.mode == "normal")

            img_fix = img.copy()

            img_fix[..., 0] -= b_diff
            img_fix[..., 2] -= r_diff

            img_fix = img_fix.astype(np.uint8)
            cv2.imwrite(params.output, img_fix)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i",
                    help="Intensity (scalar more or less around 1.0)",
                    type=float,
                    default=1.)
    ap.add_argument("-m",
                    help="Minimum brightness (positive scalar smaller 1.0)",
                    type=float,
                    default=0.)
    ap.add_argument("-r",
                    help="Blur radius (pixel)",
                    type=int,
                    default=5)
    ap.add_argument("-minred",
                    help="Minimum red:blue ratio in the fringe",
                    type=float,
                    default=0.)
    ap.add_argument("-maxred",
                    help="Maximum red:blue ratio in the fringe",
                    type=float,
                    default=.33)
    ap.add_argument("-gentle",
                    help="Gentle (Same as -m 0.8 -minred 0.15)",
                    action="store_true")
    ap.add_argument("-diff", action="store_const",
                    dest="mode",
                    const="diff",
                    default="normal",
                    help="Output image type")
    ap.add_argument("-blur", action="store_const",
                    dest="mode",
                    const="blur",
                    help="Output image type")
    ap.add_argument(help="input image",
                    dest="input")
    ap.add_argument(help="output image",
                    dest="output")
    args = ap.parse_args()
    if args.gentle:
        args.m = 0.8
        args.minred = 0.15
    img = unpurple(args)
