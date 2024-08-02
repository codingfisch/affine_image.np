import timeit
THREADS = '1'  # '1' or 'auto'
N = 10

thread_boilerplate = lambda x: '' if x == 'auto' else f'import os; os.environ["OMP_NUM_THREADS"] = "{x}"'  # default 4
markdown_table = '| Image size (Interpolation) | torch | scipy | affine-image |\n'
markdown_table += '|------|-------|-------|-------|--------------|\n'

for s in [64, 128, 256]:
    for nearest in [True, False]:
        torch_setup = f'''{thread_boilerplate(THREADS)}
import torch
import torch.nn.functional as F

im = torch.rand(1, 1, {s}, {s}, {s}).float()
affine = torch.tensor([[[1.5, 0, 0, 0],
                        [0, 1, 0, 1.0],
                        [0, 0, 1, 0]]]).float()
    '''
        mode = 'nearest' if nearest else 'bilinear'
        torch_run = f'F.grid_sample(im, F.affine_grid(affine, [1, 3, {s}, {s}, {s}], align_corners=False), mode="{mode}", align_corners=False)'
        torch_time = timeit.timeit(torch_run, setup=torch_setup, number=N) / N

        numpy_setup = f'''{thread_boilerplate(THREADS)}
import numpy as np
from scipy.ndimage import affine_transform
from affine_image import affine_transform_3d

im = np.random.rand(1, 1, {s}, {s}, {s}).astype(np.float32)
affine = np.array([[[1.5, 0, 0, 0],
                    [0, 1, 0, 1.0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]]]).astype(np.float32)
    '''
        scipy_run = f'affine_transform(im[0, 0], affine[0], output_shape=({s}, {s}, {s}), order={1 - int(nearest)})'
        scipy_time = timeit.timeit(scipy_run, setup=numpy_setup, number=N) / N
        affine_image_run = f'affine_transform_3d(im, affine, ({s}, {s}, {s}), nearest={nearest})'
        affine_image_time = timeit.timeit(affine_image_run, setup=numpy_setup, number=N) / N

        interp = 'nearest' if nearest else 'trilinear'
        markdown_table += f'| {s}³ ({interp}) | {torch_time:.3f} | {scipy_time:.3f} | {affine_image_time:.3f} |\n'

print(markdown_table)
