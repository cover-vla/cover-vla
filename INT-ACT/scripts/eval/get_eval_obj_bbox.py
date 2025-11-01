import trimesh

# Load; force='mesh' will merge a single-mesh scene into one Trimesh
mesh = trimesh.load('path/to/keyboard.glb', force='mesh')

# bounds is a (2*3) array: [ [xmin, ymin, zmin], [xmax, ymax, zmax] ]
min_corner, max_corner = mesh.bounds

xmin, ymin, zmin = min_corner
xmax, ymax, zmax = max_corner

print(f"X range: {xmin:.3f} → {xmax:.3f}")
print(f"Y range: {ymin:.3f} → {ymax:.3f}")
print(f"Z range: {zmin:.3f} → {zmax:.3f}")

# If you also want the size along each axis:
size_x, size_y, size_z = max_corner - min_corner
print(f"Size:  ΔX={size_x:.3f}, ΔY={size_y:.3f}, ΔZ={size_z:.3f}")
