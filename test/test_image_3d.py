import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from unittest import TestCase
from scipy.ndimage import affine_transform
from affine_image.image_3d import PADDINGS, affine_grid_3d, sample_nearest_3d, sample_linear_3d, affine_transform_3d
ASSERT_MARGIN = .2
SAVE_IMAGE = 'mesh'  # NOTE TO CONTRIBUTOR: Pick either None, 'brain', 'shape' or 'mesh'
SAVE_AFFINE_KEY = 'translation'  # NOTE TO CONTRIBUTOR: Pick either 'identity', 'translation', 'zoom' or 'all'
CMAPS = ('gray', 'gist_ncar')  # NOTE TO CONTRIBUTOR: Pick matplotlib colormap for visualization
# DO NOT TOUCH GLOBALS AFTER THIS LINE TO ENABLE VALID COMPARISON TO MAIN VERSION MEAN ABSOLUTE ERROR (MAE)
IN_SHAPE = (40, 109, 91)
OUT_SHAPE = (40, 100, 100)
IMAGES = ('brain', 'shapes', 'mesh')
AFFINE_DICT = {'identity': np.eye(4),
               'translation': np.array([[1, 0, 0, 0], [0, 1, 0, .5], [0, 0, 1, 0], [0, 0, 0, 1]]),
               'zoom': np.array([[1, 0, 0, 0], [0, .8, 0, 0], [0, 0, 1.2, 0], [0, 0, 0, 1]]),
               'all': np.array([[1, 0, .1, 0], [0, .7, 0, .2], [0, .1, 1.3, .0], [0, 0, 0, 1]])}
AFFINE_KEYS = tuple(AFFINE_DICT.keys())
AFFINE = np.stack(list(AFFINE_DICT.values())).astype(np.float32)


class TestImage3D(TestCase):
    def test_affine_grid(self, shape=OUT_SHAPE):
        maes = {False: 9.6e-9, True: 1.1e-8}  # UPDATE PRIOR TO NEW VERSION RELEASE
        for align_corners in [False, True]:
            grid1 = affine_grid_3d(AFFINE[:, :3].copy(), shape, align_corners)
            grid2 = F.affine_grid(torch.tensor(AFFINE)[:, :3], [len(AFFINE), 3, *shape], align_corners).numpy()
            mae = np.abs(grid1 - grid2).mean()
            main_mae = maes[align_corners]
            print_line('affine_grid_3d', f'align_corners={align_corners}', mae, main_mae)
            self.assertLess(mae, 1e-200 if main_mae == 0 else (1 + ASSERT_MARGIN) * main_mae)

    def test_resample_nearest(self, shape=OUT_SHAPE, im_shape=IN_SHAPE):
        maes = {True: {'zeros': 1.6e-4, 'border': 1.6e-4, 'reflection': 3e-4},
                False: {'zeros': 1.2e-2, 'border': 6.2e-3, 'reflection': 6.6e-3}}  # UPDATE PRIOR TO NEW VERSION RELEASE
        im = test_image(im_shape).repeat(len(AFFINE), axis=0)
        if SAVE_IMAGE is not None:
            f, axs = setup_subplots(nrows=2 * len(PADDINGS))
        for i0, align_corners in enumerate([True, False]):
            grid = affine_grid_3d(AFFINE[:, :3], shape, align_corners)
            for i1, padding in enumerate(PADDINGS):
                i = i0 * len(PADDINGS) + i1
                im1 = sample_nearest_3d(im.copy(), grid, padding, align_corners)
                im2 = F.grid_sample(torch.tensor(im), torch.tensor(grid), 'nearest', padding, align_corners).numpy()
                mae = np.abs(im1 - im2).mean()
                main_mae = maes[align_corners][padding]
                arg_string = f'align_corners={align_corners}, padding={padding}'
                if SAVE_IMAGE is not None:
                    plot_row(f, axs, i, im1, im2, title=f'{arg_string}: MAE={mae:.1e}')
                print_line('sample_nearest_3d', arg_string, mae, main_mae)
                self.assertLess(mae, 1e-200 if main_mae == 0 else (1 + ASSERT_MARGIN) * main_mae)
        if SAVE_IMAGE is not None:
            f.savefig(f'data/sample_nearest_3d_{SAVE_IMAGE}_{SAVE_AFFINE_KEY}.png')

    def test_resample_linear(self, shape=OUT_SHAPE, im_shape=IN_SHAPE):
        maes = {True: {'zeros': 6.1e-3, 'border': 3.5e-3, 'reflection': 4e-3},
                False: {'zeros': 1.6e-2, 'border': 8.4e-3, 'reflection': 9.2e-3}}  # UPDATE PRIOR TO NEW VERSION RELEASE
        im = test_image(im_shape).repeat(len(AFFINE), axis=0)
        if SAVE_IMAGE is not None:
            f, axs = setup_subplots(nrows=2 * len(PADDINGS))
        for i0, align_corners in enumerate([True, False]):
            grid = affine_grid_3d(AFFINE[:, :3], shape, align_corners)
            for i1, padding in enumerate(PADDINGS):
                i = i0 * len(PADDINGS) + i1
                im1 = sample_linear_3d(im.copy(), grid, padding, align_corners)
                im2 = F.grid_sample(torch.tensor(im), torch.tensor(grid), 'bilinear', padding, align_corners).numpy()
                mae = np.abs(im1 - im2).mean()
                main_mae = maes[align_corners][padding]
                arg_string = f'align_corners={align_corners}, padding={padding}'
                if SAVE_IMAGE is not None:
                    plot_row(f, axs, i, im1, im2, title=f'{arg_string}: MAE={mae:.1e}')
                print_line('sample_linear_3d', arg_string, mae, main_mae)
                self.assertLess(mae, 1e-200 if main_mae == 0 else (1 + ASSERT_MARGIN) * main_mae)
        if SAVE_IMAGE is not None:
            f.savefig(f'data/sample_linear_3d_{SAVE_IMAGE}_{SAVE_AFFINE_KEY}.png')

    def test_affine_transform(self, shape=OUT_SHAPE, im_shape=IN_SHAPE):
        main_mae = 1.6e-2  # UPDATE PRIOR TO NEW VERSION RELEASE
        im = test_image(im_shape).repeat(len(AFFINE), axis=0)
        im1 = affine_transform_3d(im, AFFINE, shape, align_corners=False)
        grid = F.affine_grid(torch.tensor(AFFINE)[:, :3], [len(AFFINE), 3, *shape], align_corners=False)
        im2 = F.grid_sample(torch.tensor(im), grid, align_corners=False).numpy()
        mae = np.abs(im1 - im2).mean()
        print_line('affine_transform_3d', '', mae, main_mae)
        self.assertLess(mae, 1e-200 if main_mae == 0 else (1 + ASSERT_MARGIN) * main_mae)

    def test_affine_transform_scipy(self, shape=OUT_SHAPE, im_shape=IN_SHAPE):
        maes = {True: {0: 7.3e-3, .5: 8.5e-3}, False: {0: 9.2e-3, .5: 1e-2}}  # UPDATE PRIOR TO NEW VERSION RELEASE
        affine = AFFINE.copy()
        affine = affine[:, :, [2, 1, 0, 3]][:, [2, 1, 0]]
        affine[:, :3, 3] *= np.array(im_shape) / 2
        paddings = [0, .5]
        im = test_image(im_shape).repeat(len(AFFINE), axis=0)
        if SAVE_IMAGE is not None:
            f, axs = setup_subplots(nrows=2 * len(paddings), baseline='scipy')
        for i0, nearest in enumerate([True, False]):
            for i1, padding in enumerate(paddings):
                i = i0 * len(paddings) + i1
                im1 = affine_transform_3d(im, affine, shape, nearest, padding, scipy_affine=True)
                im2 = []
                for b in range(im.shape[0]):
                    imc = []
                    for c in range(im.shape[1]):
                        im_ = affine_transform(im[b, c], affine[b], output_shape=shape, order=1 - nearest, cval=padding)
                        imc.append(im_)
                    im2.append(np.stack(imc))
                im2 = np.stack(im2)
                mae = np.abs(im1 - im2).mean()
                main_mae = maes[nearest][padding]
                arg_string = f'nearest={nearest}, padding={padding}'
                if SAVE_IMAGE is not None:
                    plot_row(f, axs, i, im1, im2, title=f'{arg_string}: MAE={mae:.1e}')
                print_line('affine_transform_3d (SCIPY)', arg_string, mae, main_mae)
                self.assertLess(mae, 1e-200 if main_mae == 0 else (1 + ASSERT_MARGIN) * main_mae)
        if SAVE_IMAGE is not None:
            f.savefig(f'data/sample_vs_scipy_3d_{SAVE_IMAGE}_{SAVE_AFFINE_KEY}.png')


def print_line(func_name, arg_string, mae, main_mae):
    color = get_print_color(mae, main_mae)
    print(f'{func_name} with {arg_string}: MAE={color}{mae:.1e} '
          f'\033[0m(<{int((1 + ASSERT_MARGIN) * 100)}% of MAE in main version {main_mae:.1e})')


def setup_subplots(nrows, tile_size=4, baseline='torch'):
    f, axs = plt.subplots(nrows, 3, figsize=(3 * tile_size, nrows * tile_size), constrained_layout=True)
    f.suptitle(f'affine-image (left column) {baseline} (middle column) affine-image - {baseline} (right column)')
    for ax in axs.flat:
        ax.axis('off')
    return f, axs


def plot_row(f, axs, row, im1, im2, title, cmaps=CMAPS):
    plane = im1.shape[2] // 2
    batch = AFFINE_KEYS.index(SAVE_AFFINE_KEY)
    axs[row, 0].imshow(im1[batch, IMAGES.index(SAVE_IMAGE), plane], vmin=0, vmax=1, cmap=cmaps[0])
    axs[row, 1].imshow(im2[batch, IMAGES.index(SAVE_IMAGE), plane], vmin=0, vmax=1, cmap=cmaps[0])
    axs[row, 1].set_title(title)
    cax = axs[row, 2].imshow((im1 - im2)[batch, IMAGES.index(SAVE_IMAGE), plane], vmin=-1, vmax=1, cmap=cmaps[1])
    f.colorbar(cax, ax=axs[row, 2])


def test_image(shape):
    return np.concatenate([brain_image(shape), shapes_image(shape), mesh_image(shape)])[None]


def brain_image(shape):
    im = np.load('data/brain_3d.npy')
    return F.interpolate(torch.from_numpy(im[None, None]), shape)[0].numpy().astype(np.float32) / 255


def shapes_image(shape):
    x, y, z = np.meshgrid(*[np.linspace(-1, 1, s) for s in shape], indexing='ij', copy=False)
    im = np.zeros_like(x, dtype=np.float32)
    cube = (np.abs(x) <= .5) & (np.abs(y) <= .9) & (np.abs(z) <= .5)
    sphere = (x ** 2 + y ** 2 + z ** 2) <= .5
    im[cube] = .5
    im[sphere] = 1
    return im[None]


def mesh_image(shape):
    x, y, z = np.meshgrid(*[np.linspace(-1, 1, s) for s in shape], indexing='ij', copy=False)
    return np.stack([z, y, x]).astype(np.float32)


def get_print_color(value0, value1, thresholds=(.5, 1. + ASSERT_MARGIN, 5, 10), colors=(22, 82, 226, 214, 196)):
    if value0 == 0:
        return f'\033[38;5;{colors[0]}m'
    for thres, color in zip(thresholds, colors):
        if value0 / value1 < thres:
            return f'\033[38;5;{color}m'
    return f'\033[38;5;{colors[-1]}m'
