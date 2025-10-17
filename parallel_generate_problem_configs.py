import h5py
import yaml
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys
import logging
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

logging.basicConfig(
    level=logging.INFO,
    format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

def extract_param_from_path(path_str, param_prefix):
    path_parts = Path(path_str).parts
    for part in path_parts:
        if part.startswith(param_prefix):
            return part[len(param_prefix):]
    return None

def read_meta(path):
    meta = {
        "path": str(path),
        "dims": 0,
        "shape": (),
        "ns": 0,
        "T": None,
        "Lx": None,
        "Ly": None,
        "Lz": None,
        "nx": None,
        "ny": None,
        "nz": None,
        "is_complex": False,
        "m_type": None,
        "c_type": None,
    }
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
                logging.warning(f"No valid u field in {path}")
                return None

            d = f[u_key]
            if d.ndim < 2:
                logging.warning(f"Invalid dimensions in {path}: ndim={d.ndim}")
                return None

            meta["ns"] = d.shape[0]
            meta["dims"] = d.ndim - 1
            meta["shape"] = d.shape[1:]
            meta["is_complex"] = is_complex

            if "/grid" in f:
                g = f["/grid"]
                for k in ["Lx", "Ly", "Lz", "nx", "ny", "nz"]:
                    if k in g.attrs:
                        meta[k] = float(g.attrs[k]) if k.startswith('L') else int(g.attrs[k])
            else:
                logging.warning(f"No /grid group in {path}")
            
            if "/time" in f:
                t = f["/time"]
                for k in ["T", "nt", "num_snapshots"]:
                    if k in t.attrs:
                        if k == "T":
                            meta["T"] = float(t.attrs[k])
                        elif k == "num_snapshots":
                            meta["ns"] = int(t.attrs[k])
                        elif k == "nt":
                            meta["nt"] = int(t.attrs[k])
            else:
                logging.warning(f"No /time group in {path}")
            
            path_str = str(path)
            meta["m_type"] = extract_param_from_path(path_str, "m_")
            meta["c_type"] = extract_param_from_path(path_str, "c_")
                
        logging.debug(f"Successfully read meta from {path}")
        return meta
    except Exception as e:
        logging.error(f"Failed to read {path}: {e}")
        return None

def infer_pde_type(basedir):
    path_str = str(basedir).lower()
    if 'kge' in path_str or 'klein_gordon' in path_str:
        return 'Klein-Gordon'
    elif 'nlse' in path_str or 'schrodinger' in path_str:
        return 'Nonlinear Schrodinger'
    elif 'kdv' in path_str:
        return 'Korteweg-de Vries'
    return 'Unknown'

def infer_boundary_type(basedir):
    path_str = str(basedir).lower()
    if 'periodic' in path_str:
        return 'periodic'
    elif 'dirichlet' in path_str:
        return 'dirichlet'
    return 'neumann'

def discover_classes_parallel(basedir):
    files = None
    
    if rank == 0:
        all_files = list(Path(basedir).rglob("*.h5"))
        from random import shuffle
        shuffle(all_files)
        files = [str(f) for f in all_files]
        logging.info(f"Found {len(files)} HDF5 files total")
    
    files = comm.bcast(files, root=0)
    
    local_files = [files[i] for i in range(len(files)) if i % size == rank]
    logging.info(f"Processing {len(local_files)} files on this rank")
    
    local_metas = []
    
    for i, fpath in enumerate(local_files):
        m = read_meta(fpath)
        if m:
            local_metas.append(m)
        else:
            logging.warning(f"Skipping invalid file: {fpath}")
        
        if (i + 1) % 10 == 0:
            logging.info(f"Processed {i+1}/{len(local_files)} files")
    
    logging.info(f"Completed processing {len(local_files)} files, extracted {len(local_metas)} valid metadata")
    
    all_metas = comm.gather(local_metas, root=0)
    
    if rank == 0:
        global_metas = []
        for meta_list in all_metas:
            global_metas.extend(meta_list)
        
        logging.info(f"Total valid files processed: {len(global_metas)}")
        return global_metas
    
    return None

def create_config_key(meta, pde_type, boundary_type):
    dims = meta["dims"]
    
    if dims == 2:
        key = (
            pde_type,
            meta.get("nx"),
            meta.get("ny"),
            None,
            meta.get("Lx"),
            meta.get("Ly"),
            None,
            meta.get("nt"),
            meta.get("ns"),
            meta.get("T"),
            boundary_type,
        )
    elif dims == 3:
        key = (
            pde_type,
            meta.get("nx"),
            meta.get("ny"),
            meta.get("nz"),
            meta.get("Lx"),
            meta.get("Ly"),
            meta.get("Lz"),
            meta.get("nt"),
            meta.get("ns"),
            meta.get("T"),
            boundary_type,
        )
    else:
        return None
    
    return key

def aggregate_configs(metas, pde_type, boundary_type):
    configs = defaultdict(lambda: {"m_classes": set(), "c_classes": set(), "count": 0})
    
    for m in metas:
        key = create_config_key(m, pde_type, boundary_type)
        if key is None:
            continue
        
        configs[key]["count"] += 1
        
        if m.get("m_type"):
            configs[key]["m_classes"].add(m["m_type"])
        if m.get("c_type"):
            configs[key]["c_classes"].add(m["c_type"])
    
    return configs

def write_config(key, data, outdir):
    pde_type, nx, ny, nz, Lx, Ly, Lz, nt, ns, T, boundary_type = key
    
    config = {
        'pde_type': pde_type,
        'boundary_condition': boundary_type,
        'num_trajectories': data["count"],
    }
    
    if ns is not None:
        config['num_snapshots'] = int(ns)
    
    if nt is not None:
        config['nt'] = int(nt)
    
    domain = {}
    if Lx is not None and Ly is not None:
        domain['Lx'] = [float(-Lx), float(Lx)]
        domain['Ly'] = [float(-Ly), float(Ly)]
    
    if Lz is not None:
        domain['Lz'] = [float(-Lz), float(Lz)]
    
    if T is not None:
        domain['T'] = [0.0, float(T)]
    
    if domain:
        config['domain'] = domain
    
    grid = {}
    if nx is not None:
        grid['nx'] = int(nx)
    if ny is not None:
        grid['ny'] = int(ny)
    if nz is not None:
        grid['nz'] = int(nz)
    
    if grid:
        config['grid'] = grid
    
    if data["m_classes"]:
        config['m_classes'] = sorted(list(data["m_classes"]))
    if data["c_classes"]:
        config['c_classes'] = sorted(list(data["c_classes"]))
    
    dims = 3 if nz is not None else 2
    shape_str = f"{nx}x{ny}" if dims == 2 else f"{nx}x{ny}x{nz}"
    
    filename = f"config_{pde_type.replace(' ', '_')}_{dims}D_{shape_str}_ns{ns}_nt{nt}.yaml"
    output_path = Path(outdir) / filename
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logging.info(f"Generated config: {output_path}")
    logging.info(f"  PDE: {pde_type}, Grid: {shape_str}, Snapshots: {ns}, Trajectories: {data['count']}")
    if data["m_classes"]:
        logging.info(f"  m classes: {sorted(list(data['m_classes']))}")
    if data["c_classes"]:
        logging.info(f"  c classes: {sorted(list(data['c_classes']))}")

def generate_config(basedir, outdir):
    if rank == 0:
        if not Path(basedir).exists():
            logging.error(f"Base directory does not exist: {basedir}")
            comm.Abort(1)
        
        try:
            Path(outdir).mkdir(parents=True, exist_ok=True)
            logging.info(f"Output directory ready: {outdir}")
        except Exception as e:
            logging.error(f"Failed to create output directory {outdir}: {e}")
            comm.Abort(1)
    
    comm.Barrier()
    
    metas = discover_classes_parallel(basedir)
    
    if rank == 0:
        if not metas:
            logging.warning(f"No valid metadata found in {basedir}")
            return
        
        pde_type = infer_pde_type(basedir)
        boundary_type = infer_boundary_type(basedir)
        logging.info(f"Inferred PDE type: {pde_type}")
        logging.info(f"Inferred boundary type: {boundary_type}")
        
        configs = aggregate_configs(metas, pde_type, boundary_type)
        
        logging.info(f"Found {len(configs)} unique configurations")
        
        for key, data in configs.items():
            try:
                write_config(key, data, outdir)
            except Exception as e:
                logging.error(f"Failed to write config for {key}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        if rank == 0:
            print("Usage: mpirun -n <nprocs> python problem_configurations.py <basedir> <outdir>")
        sys.exit(1)
    
    basedir = sys.argv[1]
    outdir = sys.argv[2]
    
    if rank == 0:
        logging.info(f"Starting configuration generation with {size} MPI ranks")
        logging.info(f"Base directory: {basedir}")
        logging.info(f"Output directory: {outdir}")
    
    generate_config(basedir, outdir)
    
    comm.Barrier()
    
    if rank == 0:
        logging.info("Configuration generation complete")
