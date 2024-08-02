import numpy as np
PADDINGS = ('zeros', 'border', 'reflection')


def affine_transform_3d(im, affine, shape, nearest=False, padding='zeros', align_corners=False, scipy_affine=False):
    align_corners = True if scipy_affine else align_corners
    theta = get_theta(affine[:, :3], im.shape[2:], shape) if scipy_affine else affine[:, :3]
    grid = affine_grid_3d(theta, shape, align_corners)
    kwargs = {'padding': padding, 'align_corners': align_corners}
    return sample_nearest_3d(im, grid, **kwargs) if nearest else sample_linear_3d(im, grid, **kwargs)


def get_theta(affine, shape, out_shape):
    size, out_size = np.array(shape), np.array(out_shape)
    theta = affine.copy()
    theta[:, :3, :3] *= out_size / size
    theta[:, :3, :3] *= out_size[None].repeat(3, 0) / out_size[:, None]
    #theta[:, :3, 3] = (1 - 1 / size) * (theta[:, :3, :3].sum(2) - 1) + 2 * theta[:, :3, 3] / size
    theta[:, :3, 3] = (1 - 1 / size) * (theta[:, :3, :3].sum(2) - 1) + 2 * theta[:, :3, 3] / (size - 1)
    return theta[:, [2, 1, 0]][:, :, [2, 1, 0, 3]]


def affine_grid_3d(theta, shape, align_corners=False):
    corners = np.ones(3) if align_corners else 1 - 1 / np.array(shape)
    grid_bins = [np.linspace(-c, c, s, dtype=theta.dtype) for c, s in zip(corners, shape)]
    x, y, z = np.meshgrid(*grid_bins, indexing='ij', copy=False)
    grid = np.stack([z, y, x, np.ones_like(x)])
    grid = grid.reshape(4, -1)
    grid = grid[None].repeat(len(theta), axis=0)
    grid = theta @ grid
    grid = grid.reshape(len(theta), 3, *shape)
    return grid.transpose(0, 2, 3, 4, 1)


def sample_linear_3d(im, grid, padding='zeros', align_corners=False):  # trilinear interpolation
    shape = np.array(im.shape[2:])
    padded_grid = pad_grid(grid, padding)
    idxs = get_indizes(padded_grid, shape, align_corners)
    mask = None if padding in ['border', 'reflection'] else pad_mask(idxs, shape, align_corners)
    idxs0 = idxs.round().astype(np.int16).clip(min=0, max=shape - 1).transpose(4, 0, 1, 2, 3)
    idxs1 = (idxs + 1).round().astype(np.int16).clip(min=0, max=shape - 1).transpose(4, 0, 1, 2, 3)
    idxs = idxs.clip(min=0, max=shape - 1).transpose(4, 0, 1, 2, 3)
    w0 = idxs1 - idxs
    w1 = 1 - w0
    b = np.arange(im.shape[0])[None, :, None, None, None]
    c = np.arange(im.shape[1])[:, None, None, None, None]
    out_im = im.transpose(1, 0, 2, 3, 4)
    out_im = (out_im[c, b, idxs0[0], idxs0[1], idxs0[2]] * w0[0] * w0[1] * w0[2] +  # 000
              out_im[c, b, idxs0[0], idxs0[1], idxs1[2]] * w0[0] * w0[1] * w1[2] +  # 001
              out_im[c, b, idxs0[0], idxs1[1], idxs0[2]] * w0[0] * w1[1] * w0[2] +  # 010
              out_im[c, b, idxs0[0], idxs1[1], idxs1[2]] * w0[0] * w1[1] * w1[2] +  # 011
              out_im[c, b, idxs1[0], idxs0[1], idxs0[2]] * w1[0] * w0[1] * w0[2] +  # 100
              out_im[c, b, idxs1[0], idxs0[1], idxs1[2]] * w1[0] * w0[1] * w1[2] +  # 101
              out_im[c, b, idxs1[0], idxs1[1], idxs0[2]] * w1[0] * w1[1] * w0[2] +  # 110
              out_im[c, b, idxs1[0], idxs1[1], idxs1[2]] * w1[0] * w1[1] * w1[2])   # 111
    if mask is not None:
        out_im[:, mask] = 0 if padding == 'zeros' else padding
    out_im = out_im.clip(im.min(), im.max())
    return out_im.transpose(1, 0, 2, 3, 4)


def sample_nearest_3d(im, grid, padding='zeros', align_corners=False):
    shape = np.array(im.shape[2:])
    padded_grid = pad_grid(grid, padding)
    idxs = get_indizes(padded_grid, shape, align_corners)
    mask = None if padding in ['border', 'reflection'] else pad_mask(idxs, shape, align_corners)
    idxs = idxs.round().astype(np.int16).clip(0, shape - 1)
    idxs = idxs.transpose(4, 0, 1, 2, 3)
    b = np.arange(im.shape[0])[None, :, None, None, None]
    c = np.arange(im.shape[1])[:, None, None, None, None]
    out_im = im.transpose(1, 0, 2, 3, 4)
    out_im = out_im[c, b, idxs[0], idxs[1], idxs[2]]
    if mask is not None:
        out_im[:, mask] = 0 if padding == 'zeros' else padding
    return out_im.transpose(1, 0, 2, 3, 4)


def pad_grid(grid, padding):
    assert padding in PADDINGS or isinstance(padding, (int, float)), f'padding must be int, float or in {PADDINGS})'
    if padding == 'reflection':
        reflect_grid = grid.copy()
        over, under = reflect_grid > 1, reflect_grid < -1
        reflect_grid[over] = 2 - reflect_grid[over]
        reflect_grid[under] = -2 - reflect_grid[under]
        return reflect_grid
    else:
        return grid


def pad_mask(idxs, shape, align_corners):
    offset = 0 if align_corners else 1 / shape[::-1]
    return (idxs < -.5 - offset).any(axis=-1) | (idxs > (shape - .5 + offset)).any(axis=-1)


def get_indizes(grid, shape, align_corners):
    align = 1 if align_corners else 1 - 1 / shape[::-1]
    return (grid[..., ::-1] / align + 1) * .5 * (shape - 1)
