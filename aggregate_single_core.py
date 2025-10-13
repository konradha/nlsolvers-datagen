from scripts.downsampling import downsample_interpolation as downsample_2d
from scripts.downsampling import downsample_interpolation_3d as downsample_3d

import h5py
import numpy as np
import logging
import time
from pathlib import Path
from collections import defaultdict
from netCDF4 import Dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)


def read_meta(path):
    meta = {"path": str(path), "dims": 0, "shape": (), "ns": 0, "Lx": None, "Ly": None, "Lz": None, "is_complex": False}
    try:
        with h5py.File(path, "r") as f:
            is_complex = False
            u_key = None

            if "/re_u" in f:
                is_complex = True
                u_key = "/re_u"
            elif "/u" in f:
                u_key = "/u"
                d = f["/u"]
                is_complex = np.issubdtype(d.dtype, np.complexfloating)
            else:
                return None

            d = f[u_key]
            if d.ndim < 2:
                return None

            meta["ns"] = d.shape[0]
            meta["dims"] = d.ndim - 1
            meta["shape"] = d.shape[1:]
            meta["is_complex"] = is_complex

            if "/grid" in f:
                g = f["/grid"]
                for k in ["Lx", "Ly", "Lz"]:
                    if k in g.attrs:
                        meta[k] = float(g.attrs[k])
        return meta
    except Exception as e:
        logging.warning(f"Failed to read {path}: {e}")
        return None


def discover_classes(basedir):
    from random import shuffle
    files = list(Path(basedir).rglob("*.h5"))
    shuffle(files)
    classes = defaultdict(list)
    metas = {}

    logging.info(f"Scanning {len(files)} HDF5 files")

    for i, f in enumerate(files):
        m = read_meta(f)
        if not m:
            continue

        key = (
            m['dims'],
            tuple(m["shape"]),
            m["ns"],
            tuple(v for v in [m["Lx"], m["Ly"], m["Lz"]] if v is not None),
            m["is_complex"],
        )
        classes[key].append(str(f))
        metas[str(f)] = m

        if (i + 1) % 100 == 0:
            logging.info(f"Scanned {i+1}/{len(files)} files")

    logging.info(f"Completed scanning {len(files)} files")
    return classes, metas


def downsample_field_2d(field, original_shape, target_shape, Lx, Ly):
    if tuple(original_shape) == tuple(target_shape):
        return field

    if field.ndim == 2:
        field = field[np.newaxis, ...]
        squeeze = True
    else:
        squeeze = False

    result = downsample_2d(field, target_shape, Lx, Ly)

    if squeeze:
        result = result[0]

    return result


def downsample_field_3d(field, original_shape, target_shape, Lx, Ly, Lz):
    if tuple(original_shape) == tuple(target_shape):
        return field

    if field.ndim == 3:
        field = field[np.newaxis, ...]
        squeeze = True
    else:
        squeeze = False

    result = downsample_3d(field, target_shape, Lx, Ly, Lz)

    if squeeze:
        result = result[0]

    return result


def read_dataset(f, path, shape, target_shape, Lx, Ly, Lz, dims, is_complex_storage):
    if path not in f:
        return None

    raw_data = f[path]
    is_complex_dtype = np.issubdtype(raw_data.dtype, np.complexfloating)

    if is_complex_dtype and is_complex_storage:
        arr = np.asarray(raw_data)
        arr_real = arr.real.astype(np.float32)
        arr_imag = arr.imag.astype(np.float32)

        if arr_real.size == 0 or arr_imag.size == 0:
            return None, None

        if np.any(np.isnan(arr_real)) or np.any(np.isnan(arr_imag)):
            return None, None

        if tuple(shape) != tuple(target_shape):
            if dims == 2:
                arr_real = downsample_field_2d(arr_real, shape, target_shape, Lx, Ly)
                arr_imag = downsample_field_2d(arr_imag, shape, target_shape, Lx, Ly)
            elif dims == 3:
                arr_real = downsample_field_3d(arr_real, shape, target_shape, Lx, Ly, Lz)
                arr_imag = downsample_field_3d(arr_imag, shape, target_shape, Lx, Ly, Lz)

        return arr_real, arr_imag
    else:
        arr = np.asarray(raw_data, dtype=np.float32)
        if arr.size == 0:
            return None

        if np.any(np.isnan(arr)):
            return None

        if tuple(shape) != tuple(target_shape):
            if dims == 2:
                arr = downsample_field_2d(arr, shape, target_shape, Lx, Ly)
            elif dims == 3:
                arr = downsample_field_3d(arr, shape, target_shape, Lx, Ly, Lz)

        return arr


def write_class_to_nc(key, files, metas, outdir, target_shape=None):
    dims, shape, ns, L, is_complex = key
    shape = tuple(shape)
    if target_shape is None:
        target_shape = shape

    nt = len(files)
    spat = int(np.prod(target_shape))
    temp = ns * spat

    out_path = Path(outdir) / f"class_{dims}D_shape_{'x'.join(map(str,target_shape))}_ns_{ns}_{'complex' if is_complex else 'real'}.nc"
    logging.info(f"Writing {out_path.name} from {nt} trajectories")

    nc = Dataset(out_path, "w", format="NETCDF4")
    nc.createDimension("total_temporal", nt * temp)
    nc.createDimension("total_spatial", nt * spat)

    if is_complex:
        v_re_u = nc.createVariable("re_u", "f4", ("total_temporal",) , chunksizes=(65536,))
        v_im_u = nc.createVariable("im_u", "f4", ("total_temporal",) , chunksizes=(65536,))
        v_re_u0 = nc.createVariable("re_u0", "f4", ("total_spatial",), chunksizes=(65536,))
        v_im_u0 = nc.createVariable("im_u0", "f4", ("total_spatial",), chunksizes=(65536,))
    else:
        v_u = nc.createVariable("u", "f4", ("total_temporal",) , chunksizes=(65536,))
        v_v = nc.createVariable("v", "f4", ("total_temporal",) , chunksizes=(65536,))
        v_u0 = nc.createVariable("u0", "f4", ("total_spatial",), chunksizes=(65536,))
        v_v0 = nc.createVariable("v0", "f4", ("total_spatial",), chunksizes=(65536,))

    v_m = nc.createVariable("m", "f4", ("total_spatial",))
    v_c = nc.createVariable("c", "f4", ("total_spatial",))

    nc.num_snapshots = np.int32(ns)
    nc.num_trajectories = np.int32(nt)
    nc.spatial_shape = str(target_shape)
    nc.Lx = np.float32(L[0])
    nc.Ly = np.float32(L[1])
    if len(L) == 3:
        nc.Lz = np.float32(L[2])
    nc.is_complex = np.int32(1 if is_complex else 0)
    nc.dims = np.int32(dims)

    off_t = 0
    off_s = 0
    skipped = 0
    total_read_time = 0.0
    total_ic_time = 0.0

    for idx, fpath in enumerate(files):
        t_file_start = time.perf_counter()
        m = metas[fpath]
        Lx, Ly, Lz = m["Lx"], m["Ly"], m.get("Lz")

        try:
            with h5py.File(fpath, "r") as f:
                t_read_start = time.perf_counter()

                if is_complex:
                    if "/re_u" in f:
                        re_u = read_dataset(f, "/re_u", m["shape"], target_shape, Lx, Ly, Lz, m["dims"], False)
                        im_u = read_dataset(f, "/im_u", m["shape"], target_shape, Lx, Ly, Lz, m["dims"], False)
                    else:
                        result = read_dataset(f, "/u", m["shape"], target_shape, Lx, Ly, Lz, m["dims"], True)
                        if isinstance(result, tuple):
                            re_u, im_u = result
                        else:
                            re_u = im_u = None

                    if re_u is None or im_u is None:
                        skipped += 1
                        logging.warning(f"  [{idx+1}/{nt}] Skipped {Path(fpath).name} (NaN or missing data)")
                        off_t += temp
                        off_s += spat
                        continue

                    logging.debug(f"[WRITE ] re u: off_t={off_t}, u.size={re_u.size}, slice={off_t}:{off_t+re_u.size}")
                    v_re_u[off_t:off_t + re_u.size] = np.ascontiguousarray(re_u.flatten())

                    logging.debug(f"[WRITE ] im u: off_t={off_t}, u.size={im_u.size}, slice={off_t}:{off_t+im_u.size}")
                    v_im_u[off_t:off_t + im_u.size] =  np.ascontiguousarray(im_u.flatten())
                    del re_u, im_u
                else:
                    u = read_dataset(f, "/u", m["shape"], target_shape, Lx, Ly, Lz, m["dims"], False)
                    v = read_dataset(f, "/v", m["shape"], target_shape, Lx, Ly, Lz, m["dims"], False)

                    if u is None or v is None:
                        skipped += 1
                        logging.warning(f"  [{idx+1}/{nt}] Skipped {Path(fpath).name} (NaN or missing data)")
                        off_t += temp
                        off_s += spat
                        continue

                    logging.info(f"[WRITE] u: off_t={off_t}, u.size={u.size}, slice={off_t}:{off_t+u.size}")
                    v_u[off_t:off_t + u.size] = np.ascontiguousarray(u.flatten())
                    v_v[off_t:off_t + v.size] = np.ascontiguousarray(v.flatten())
                    del u, v

                t_read_elapsed = time.perf_counter() - t_read_start
                total_read_time += t_read_elapsed

                t_ic_start = time.perf_counter()

                if "/initial_condition" in f:
                    g = f["/initial_condition"]

                    if is_complex:
                        if "re_u0" in g:
                            re_u0 = np.asarray(g["re_u0"], dtype=np.float32)
                            u0_shape = re_u0.shape
                            if m["dims"] == 2:
                                re_u0 = downsample_field_2d(re_u0, u0_shape, target_shape, Lx, Ly)
                            elif m["dims"] == 3:
                                re_u0 = downsample_field_3d(re_u0, u0_shape, target_shape, Lx, Ly, Lz)
                            v_re_u0[off_s:off_s + re_u0.size] = np.ascontiguousarray(re_u0.flatten())
                            del re_u0
                        elif "u0" in g:
                            u0_raw = g["u0"]
                            if np.issubdtype(u0_raw.dtype, np.complexfloating):
                                u0_arr = np.ascontiguousarray(np.asarray(u0_raw))
                                u0_shape = u0_arr.shape
                                re_u0 = np.ascontiguousarray(u0_arr.real.astype(np.float32))
                                im_u0 = np.ascontiguousarray(u0_arr.imag.astype(np.float32))
                                if m["dims"] == 2:
                                    re_u0 = downsample_field_2d(re_u0, u0_shape, target_shape, Lx, Ly)
                                    im_u0 = downsample_field_2d(im_u0, u0_shape, target_shape, Lx, Ly)
                                elif m["dims"] == 3:
                                    re_u0 = downsample_field_3d(re_u0, u0_shape, target_shape, Lx, Ly, Lz)
                                    im_u0 = downsample_field_3d(im_u0, u0_shape, target_shape, Lx, Ly, Lz)
                                v_re_u0[off_s:off_s + re_u0.size] = np.ascontiguousarray(re_u0.flatten())
                                v_im_u0[off_s:off_s + im_u0.size] = np.ascontiguousarray(im_u0.flatten())
                                del u0_arr, re_u0, im_u0

                        if "im_u0" in g:
                            im_u0 = np.asarray(g["im_u0"], dtype=np.float32)
                            u0_shape = im_u0.shape
                            if m["dims"] == 2:
                                im_u0 = downsample_field_2d(im_u0, u0_shape, target_shape, Lx, Ly)
                            elif m["dims"] == 3:
                                im_u0 = downsample_field_3d(im_u0, u0_shape, target_shape, Lx, Ly, Lz)
                            v_im_u0[off_s:off_s + im_u0.size] = im_u0.flatten()
                            del im_u0
                    else:
                        if "u0" in g:
                            u0 = np.asarray(g["u0"], dtype=np.float32)
                            u0_shape = u0.shape
                            if m["dims"] == 2:
                                u0 = downsample_field_2d(u0, u0_shape, target_shape, Lx, Ly)
                            elif m["dims"] == 3:
                                u0 = downsample_field_3d(u0, u0_shape, target_shape, Lx, Ly, Lz)
                            v_u0[off_s:off_s + u0.size] = np.ascontiguousarray(u0.flatten())
                            del u0

                        if "v0" in g:
                            v0 = np.asarray(g["v0"], dtype=np.float32)
                            v0_shape = v0.shape
                            if m["dims"] == 2:
                                v0 = downsample_field_2d(v0, v0_shape, target_shape, Lx, Ly)
                            elif m["dims"] == 3:
                                v0 = downsample_field_3d(v0, v0_shape, target_shape, Lx, Ly, Lz)
                            v_v0[off_s:off_s + v0.size] = np.ascontiguousarray(v0.flatten())
                            del v0

                mset = ["/m", "/focusing/m"]
                cset = ["/c", "/focusing/c", "/anisotropy/c"]

                for p in mset:
                    if p in f:
                        param_shape = f[p].shape
                        arr = read_dataset(f, p, param_shape, target_shape, Lx, Ly, Lz, m["dims"], False)
                        if arr is not None:
                            v_m[off_s:off_s + arr.size] = np.ascontiguousarray(arr.flatten())
                            del arr
                            break

                for p in cset:
                    if p in f:
                        param_shape = f[p].shape
                        arr = read_dataset(f, p, param_shape, target_shape, Lx, Ly, Lz, m["dims"], False)
                        if arr is not None:
                            v_c[off_s:off_s + arr.size] = np.ascontiguousarray(arr.flatten())
                            del arr
                            break

                t_ic_elapsed = time.perf_counter() - t_ic_start
                total_ic_time += t_ic_elapsed

            off_t += temp
            off_s += spat

            t_file_elapsed = time.perf_counter() - t_file_start
            logging.info(f"  [{idx+1}/{nt}] {Path(fpath).name} (read: {t_read_elapsed:.2f}s, IC/params: {t_ic_elapsed:.2f}s, total: {t_file_elapsed:.2f}s)")

            if (idx + 1) % 10 == 0:
                nc.sync()

        except Exception as e:
            off_t += temp
            off_s += spat
            import traceback as tb

            logging.error(f"""

            {tb.format_exc()}

            """)
            logging.error(f"  [{idx+1}/{nt}] Failed to process {Path(fpath).name}: {e}")

            skipped += 1
            continue

    nc.close()

    avg_read_time = total_read_time / max(1, nt - skipped)
    avg_ic_time = total_ic_time / max(1, nt - skipped)

    logging.info(f"  Completed {out_path.name}")
    logging.info(f"  Average read time: {avg_read_time:.2f}s, average IC/params time: {avg_ic_time:.2f}s")
    if skipped > 0:
        logging.warning(f"  Skipped {skipped}/{nt} trajectories")


def main(basedir, outdir):
    t_total_start = time.perf_counter()

    Path(outdir).mkdir(parents=True, exist_ok=True)
    classes, metas = discover_classes(basedir)

    logging.info(f"Discovered {len(classes)} parameter classes")

    for k, v in classes.items():
        dims, shape, ns, L, is_complex = k
        logging.info(f"{dims}D shape={shape} ns={ns} L={L} complex={is_complex} count={len(v)}")
        write_class_to_nc(k, v, metas, outdir)

    t_total_elapsed = time.perf_counter() - t_total_start
    logging.info(f"Aggregation complete in {t_total_elapsed:.1f}s ({t_total_elapsed/3600:.2f}h)")


if __name__ == "__main__":
    import sys
    base = str(sys.argv[1])
    outdir = str(sys.argv[2])
    main(base, outdir)
