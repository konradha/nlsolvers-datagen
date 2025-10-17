import netCDF4 as nc
import sys
from pathlib import Path

def dump_netcdf_header(filepath):
    print(f"NetCDF Header Dump for: {filepath.name}\n")
    print("```")
    
    with nc.Dataset(filepath, 'r') as ds:
        print(f"netcdf {filepath.stem} {{")
        print("dimensions:")
        for dim_name, dim in ds.dimensions.items():
            size = len(dim) if not dim.isunlimited() else "UNLIMITED"
            print(f"    {dim_name} = {size} ;")
        
        print("\nvariables:")
        for var_name, var in ds.variables.items():
            dims_str = ', '.join(var.dimensions)
            print(f"    {var.dtype} {var_name}({dims_str}) ;")
            for attr_name in var.ncattrs():
                attr_val = getattr(var, attr_name)
                print(f"        {var_name}:{attr_name} = {attr_val} ;")
        
        print("\n// global attributes:")
        for attr_name in ds.ncattrs():
            attr_val = getattr(ds, attr_name)
            print(f"    :{attr_name} = {attr_val} ;")
        
        print("}")
    
    print("```\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python dump_header.py <netcdf_file>")
        sys.exit(1)
    
    filepath = Path(sys.argv[1])
    
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    
    if filepath.suffix != '.nc':
        print(f"Error: Not a NetCDF file: {filepath}")
        sys.exit(1)
    
    dump_netcdf_header(filepath)
