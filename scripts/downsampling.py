import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator


from sys import argv


def downsample_fft(u: np.ndarray, target_shape: tuple) -> np.ndarray:
    assert len(u.shape) == 3
    assert len(target_shape) == 2
    is_complex = np.iscomplexobj(u)
    original_shape = u.shape[-2:]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dtype = torch.complex128 if is_complex else torch.float64
    u_tensor = torch.from_numpy(u).to(device=device, dtype=dtype)

    ft = torch.fft.fft2(u_tensor, norm='ortho', dim=(-2, -1))
    ft_shifted = torch.fft.fftshift(ft, dim=(-2, -1))
    start_x = (original_shape[0] - target_shape[0]) // 2
    start_y = (original_shape[1] - target_shape[1]) // 2
    ft_cropped = ft_shifted[..., start_x:start_x + target_shape[0],
                            start_y:start_y + target_shape[1]]
    ft_cropped = torch.fft.ifftshift(ft_cropped, dim=(-2, -1))
    downsampled = torch.fft.ifft2(ft_cropped, norm='ortho', dim=(-2, -1))

    downsampled_np = downsampled.cpu().numpy()
    if not is_complex:
        downsampled_np = downsampled_np.real

    return downsampled_np.astype(u.dtype)


def reconstruct_fft(downsampled: np.ndarray,
                    original_shape: tuple) -> np.ndarray:
    assert len(downsampled.shape) == 3
    assert len(original_shape) == 2

    is_complex = np.iscomplexobj(downsampled)
    current_shape = downsampled.shape[-2:]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dtype = torch.complex128 if is_complex else torch.float64
    ds_tensor = torch.from_numpy(downsampled).to(device=device, dtype=dtype)
    ft = torch.fft.fft2(ds_tensor, norm='ortho', dim=(-2, -1))
    ft_shifted = torch.fft.fftshift(ft, dim=(-2, -1))
    padded_ft = torch.zeros(
        (*ds_tensor.shape[:-2], original_shape[0], original_shape[1]),
        dtype=dtype,
        device=device
    )
    start_x = (original_shape[0] - current_shape[0]) // 2
    start_y = (original_shape[1] - current_shape[1]) // 2
    padded_ft[...,
              start_x:start_x + current_shape[0],
              start_y:start_y + current_shape[1]] = ft_shifted

    padded_ft = torch.fft.ifftshift(padded_ft, dim=(-2, -1))
    reconstructed = torch.fft.ifft2(padded_ft, norm='ortho', dim=(-2, -1))
    reconstructed_np = reconstructed.cpu().numpy()
    if not is_complex:
        reconstructed_np = reconstructed_np.real
    return reconstructed_np.astype(u.dtype)


def downsample_interpolation(
    u: np.ndarray,
    target_shape,
    Lx: float,
    Ly: float,
    original_grid=None
) -> np.ndarray:

    assert len(u.shape) == 3, "Input data should be 3D (nt, nx, ny)"
    assert len(
        target_shape) == 2, "Target shape should be 2D (target_nx, target_ny)"

    nt, nx, ny = u.shape
    target_nx, target_ny = target_shape
    if original_grid is None:
        x = np.linspace(-Lx, Lx, nx)
        y = np.linspace(-Ly, Ly, ny)
    else:
        x, y = original_grid

    x_new = np.linspace(-Lx, Lx, target_nx)
    y_new = np.linspace(-Ly, Ly, target_ny)
    downsampled = np.zeros((nt, target_nx, target_ny), dtype=u.dtype)
    X_new, Y_new = np.meshgrid(x_new, y_new, indexing='ij')
    for t in range(nt):
        interp_func = RegularGridInterpolator(
            (x, y),
            u[t],
            method='linear',
            bounds_error=False,
            fill_value=None
        )

        points = np.vstack([X_new.ravel(), Y_new.ravel()]).T
        downsampled[t] = interp_func(points).reshape(target_nx, target_ny)
    return downsampled.astype(u.dtype)


def reconstruct_interpolation(
    downsampled: np.ndarray,
    original_shape,
    Lx: float,
    Ly: float,
    downsampled_grid=None
) -> np.ndarray:

    assert len(
        downsampled.shape) == 3, "Input data should be 3D (nt, target_nx, target_ny)"
    assert len(original_shape) == 2, "Original shape should be 2D (nx, ny)"

    nt, target_nx, target_ny = downsampled.shape
    nx, ny = original_shape

    if downsampled_grid is None:
        x_ds = np.linspace(-Lx, Lx, target_nx)
        y_ds = np.linspace(-Ly, Ly, target_ny)
    else:
        x_ds, y_ds = downsampled_grid

    x = np.linspace(-Lx, Lx, nx)
    y = np.linspace(-Ly, Ly, ny)

    reconstructed = np.zeros((nt, nx, ny), dtype=downsampled.dtype)

    X, Y = np.meshgrid(x, y, indexing='ij')
    for t in range(nt):
        interp_func = RegularGridInterpolator(
            (x_ds, y_ds),
            downsampled[t],
            method='linear',
            bounds_error=False,
            fill_value=None
        )

        points = np.vstack([X.ravel(), Y.ravel()]).T
        reconstructed[t] = interp_func(points).reshape(nx, ny)
    return reconstructed.astype(u.dtype)


def downsample_fft_3d(u: np.ndarray, target_shape: tuple) -> np.ndarray:
    assert len(u.shape) == 4, "Input data should be 4D (nt, nx, ny, nz)"
    assert len(
        target_shape) == 3, "Target shape should be 3D (target_nx, target_ny, target_nz)"

    is_complex = np.iscomplexobj(u)
    original_shape = u.shape[-3:]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dtype = torch.complex128 if is_complex else torch.float64
    u_tensor = torch.from_numpy(u).to(device=device, dtype=dtype)
    ft = torch.fft.fftn(u_tensor, dim=(-3, -2, -1), norm='ortho')
    ft_shifted = torch.fft.fftshift(ft, dim=(-3, -2, -1))
    start_x = (original_shape[0] - target_shape[0]) // 2
    start_y = (original_shape[1] - target_shape[1]) // 2
    start_z = (original_shape[2] - target_shape[2]) // 2
    ft_cropped = ft_shifted[...,
                            start_x:start_x + target_shape[0],
                            start_y:start_y + target_shape[1],
                            start_z:start_z + target_shape[2]]
    ft_cropped = torch.fft.ifftshift(ft_cropped, dim=(-3, -2, -1))
    downsampled = torch.fft.ifftn(ft_cropped, dim=(-3, -2, -1), norm='ortho')

    downsampled_np = downsampled.cpu().numpy()
    if not is_complex:
        downsampled_np = downsampled_np.real

    return downsampled_np.astype(u.dtype)


def reconstruct_fft_3d(downsampled: np.ndarray,
                       original_shape: tuple) -> np.ndarray:
    assert len(
        downsampled.shape) == 4, "Input data should be 4D (nt, target_nx, target_ny, target_nz)"
    assert len(original_shape) == 3, "Original shape should be 3D (nx, ny, nz)"

    is_complex = np.iscomplexobj(downsampled)
    current_shape = downsampled.shape[-3:]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dtype = torch.complex128 if is_complex else torch.float64
    ds_tensor = torch.from_numpy(downsampled).to(device=device, dtype=dtype)
    ft = torch.fft.fftn(ds_tensor, dim=(-3, -2, -1), norm='ortho')
    ft_shifted = torch.fft.fftshift(ft, dim=(-3, -2, -1))
    padded_ft = torch.zeros(
        (*ds_tensor.shape[:-3], original_shape[0],
         original_shape[1], original_shape[2]),
        dtype=dtype,
        device=device
    )
    start_x = (original_shape[0] - current_shape[0]) // 2
    start_y = (original_shape[1] - current_shape[1]) // 2
    start_z = (original_shape[2] - current_shape[2]) // 2
    padded_ft[...,
              start_x:start_x + current_shape[0],
              start_y:start_y + current_shape[1],
              start_z:start_z + current_shape[2]] = ft_shifted

    padded_ft = torch.fft.ifftshift(padded_ft, dim=(-3, -2, -1))
    reconstructed = torch.fft.ifftn(padded_ft, dim=(-3, -2, -1), norm='ortho')

    reconstructed_np = reconstructed.cpu().numpy()
    if not is_complex:
        reconstructed_np = reconstructed_np.real

    return reconstructed_np.astype(u.dtype)


def downsample_interpolation_3d(
    u: np.ndarray,
    target_shape: tuple,
    Lx: float,
    Ly: float,
    Lz: float,
    original_grid=None
) -> np.ndarray:
    assert len(u.shape) == 4, "Input data should be 4D (nt, nx, ny, nz)"
    assert len(
        target_shape) == 3, "Target shape should be 3D (target_nx, target_ny, target_nz)"

    nt, nx, ny, nz = u.shape
    target_nx, target_ny, target_nz = target_shape

    if original_grid is None:
        x = np.linspace(-Lx, Lx, nx)
        y = np.linspace(-Ly, Ly, ny)
        z = np.linspace(-Lz, Lz, nz)
    else:
        x, y, z = original_grid

    x_new = np.linspace(-Lx, Lx, target_nx)
    y_new = np.linspace(-Ly, Ly, target_ny)
    z_new = np.linspace(-Lz, Lz, target_nz)
    downsampled = np.zeros(
        (nt, target_nx, target_ny, target_nz), dtype=u.dtype)
    X_new, Y_new, Z_new = np.meshgrid(x_new, y_new, z_new, indexing='ij')
    points_new = np.vstack([X_new.ravel(), Y_new.ravel(), Z_new.ravel()]).T

    for t in range(nt):
        interp_func = RegularGridInterpolator(
            (x, y, z),
            u[t],
            method='linear',
            bounds_error=False,
            fill_value=None
        )

        downsampled[t] = interp_func(points_new).reshape(
            target_nx, target_ny, target_nz)

    return downsampled.astype(u.dtype)


def reconstruct_interpolation_3d(
    downsampled: np.ndarray,
    original_shape: tuple,
    Lx: float,
    Ly: float,
    Lz: float,
    downsampled_grid=None
) -> np.ndarray:
    assert len(
        downsampled.shape) == 4, "Input data should be 4D (nt, target_nx, target_ny, target_nz)"
    assert len(original_shape) == 3, "Original shape should be 3D (nx, ny, nz)"

    nt, target_nx, target_ny, target_nz = downsampled.shape
    nx, ny, nz = original_shape

    if downsampled_grid is None:
        x_ds = np.linspace(-Lx, Lx, target_nx)
        y_ds = np.linspace(-Ly, Ly, target_ny)
        z_ds = np.linspace(-Lz, Lz, target_nz)
    else:
        x_ds, y_ds, z_ds = downsampled_grid

    x = np.linspace(-Lx, Lx, nx)
    y = np.linspace(-Ly, Ly, ny)
    z = np.linspace(-Lz, Lz, nz)

    reconstructed = np.zeros((nt, nx, ny, nz), dtype=downsampled.dtype)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    for t in range(nt):
        interp_func = RegularGridInterpolator(
            (x_ds, y_ds, z_ds),
            downsampled[t],
            method='linear',
            bounds_error=False,
            fill_value=None
        )

        reconstructed[t] = interp_func(points).reshape(nx, ny, nz)

    return reconstructed.astype(u.dtype)


def main_2d():
    fname_u = str(argv[1])
    target_nx, target_ny = int(argv[2]), int(argv[3])

    orig_nx, orig_ny = int(argv[4]), int(argv[5])
    Lx, Ly = float(argv[6]), float(argv[7])


    u = np.load(fname_u)
    assert len(u.shape) == 3
    nt = u.shape[0]
    nx, ny = u.shape[1], u.shape[2]

    ts = np.random.randint(0, nt)

    is_complex = np.iscomplexobj(u)
    downsampled = downsample_fft(u, (target_nx, target_ny))
    reconstructed = reconstruct_fft(downsampled, (nx, ny))
    #downsampled = downsample_interpolation(
    #    u, (target_nx, target_ny), Lx, Ly)
    #reconstructed = reconstruct_interpolation(
    #    downsampled, (u.shape[1], u.shape[2]), Lx, Ly)



    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
    im = ax1.imshow(np.real(u[ts]) if is_complex else u[ts])
    ax1.set_title("Original")
    ax2.imshow(np.real(downsampled[ts]) if is_complex else downsampled[ts])
    ax2.set_title("Downsampled")

    diff = np.abs(np.real(u[ts] - reconstructed[ts])) if is_complex else np.abs(u[ts] - reconstructed[ts])
    print(f"{is_complex=} {np.linalg.norm(u[ts] - reconstructed[ts], ord='fro')=:.4f}")
    im_r = ax3.imshow(diff, cmap="seismic")
    ax3.set_title("Reconstructed to original diff")



    fig.colorbar(im, ax=[ax1, ax2], label='magnitude / [1]')
    fig.colorbar(im_r, ax=[ax3], label="Diff / [1]")
    fig.suptitle(f"{'NLSE' if is_complex else 'SGE'}; sample: {ts} / {nt} (num snapshots)")

    plt.show()


def main_3d():
    from sys import argv
    import matplotlib.pyplot as plt

    fname_u = str(argv[1])
    target_nx, target_ny, target_nz = int(argv[2]), int(argv[3]), int(argv[4])
    orig_nx, orig_ny, orig_nz = int(argv[5]), int(argv[6]), int(argv[7])
    Lx, Ly, Lz = float(argv[8]), float(argv[9]), float(argv[10])

    u = np.load(fname_u)
    assert len(u.shape) == 4
    nt = u.shape[0]
    nx, ny, nz = u.shape[1], u.shape[2], u.shape[3]

    ts = np.random.randint(0, nt)
    slice_z = nz // 2

    is_complex = np.iscomplexobj(u)

    use_fft = False
    if use_fft:
        downsampled = downsample_fft_3d(u, (target_nx, target_ny, target_nz))
        reconstructed = reconstruct_fft_3d(downsampled, (nx, ny, nz))
    else:
        downsampled = downsample_interpolation_3d(
            u, (target_nx, target_ny, target_nz), Lx, Ly, Lz)
        reconstructed = reconstruct_interpolation_3d(
            downsampled, (nx, ny, nz), Lx, Ly, Lz)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    if is_complex:
        im1 = ax1.imshow(np.real(u[ts, :, :, slice_z]))
    else:
        im1 = ax1.imshow(u[ts, :, :, slice_z])
    ax1.set_title("Original (z-slice)")

    ds_slice_z = slice_z * target_nz // nz
    if is_complex:
        im2 = ax2.imshow(np.real(downsampled[ts, :, :, ds_slice_z]))
    else:
        im2 = ax2.imshow(downsampled[ts, :, :, ds_slice_z])
    ax2.set_title("Downsampled (z-slice)")

    if is_complex:
        diff = np.abs(np.real(u[ts, :, :, slice_z] -
                      reconstructed[ts, :, :, slice_z]))
    else:
        diff = np.abs(u[ts, :, :, slice_z] - reconstructed[ts, :, :, slice_z])
    im3 = ax3.imshow(diff, cmap="seismic")
    ax3.set_title("Reconstruction Error (z-slice)")

    fig.colorbar(im1, ax=[ax1, ax2], label='magnitude / [1]')
    fig.colorbar(im3, ax=[ax3], label="Diff / [1]")

    method = "FFT" if use_fft else "Interpolation"
    error_norm = np.linalg.norm(u[ts] - reconstructed[ts], ord='fro')
    fig.suptitle(f"3D {'NLSE' if is_complex else 'SGE'} Downsampling ({method})\n" +
                 f"Sample: {ts}/{nt}, Error norm: {error_norm:.4f}")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    pass 
