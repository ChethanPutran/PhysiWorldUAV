
from pathlib import Path
import re

import numpy as np
import trimesh
from pxr import Usd, UsdGeom


USD_FILE = Path("data/hummingbird.usd")
OUTPUT_DIR = Path("stl_parts")


def safe_name(path):
    name = path.strip("/").replace("/", "__")
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name) or "mesh"


def export_mesh(mesh_prim, stage, output_dir):
    mesh = UsdGeom.Mesh(mesh_prim)

    points = mesh.GetPointsAttr().Get()
    face_counts = mesh.GetFaceVertexCountsAttr().Get()
    face_indices = mesh.GetFaceVertexIndicesAttr().Get()

    if not points or not face_counts or not face_indices:
        print(f"Skipping empty mesh: {mesh_prim.GetPath()}")
        return

    # Transform vertices into world coordinates.
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    world_transform = xform_cache.GetLocalToWorldTransform(mesh_prim)

    vertices = np.array([
        tuple(world_transform.Transform(p))
        for p in points
    ], dtype=np.float64)

    # Triangulate polygon faces using a fan.
    # Suitable for triangles and typical convex polygons.
    triangles = []
    offset = 0

    for count in face_counts:
        face = list(face_indices[offset:offset + count])
        offset += count

        for i in range(1, len(face) - 1):
            triangles.append([face[0], face[i], face[i + 1]])

    if not triangles:
        print(f"No triangles: {mesh_prim.GetPath()}")
        return

    faces = np.asarray(triangles, dtype=np.int64)

    # Respect USD's authored face orientation.
    if mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
        faces = faces[:, [0, 2, 1]]

    result = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )

    output_path = output_dir / f"{safe_name(str(mesh_prim.GetPath()))}.stl"
    result.export(output_path)

    print(f"Exported: {output_path} ({len(faces)} triangles)")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.Open(str(USD_FILE))
    if stage is None:
        raise RuntimeError(f"Could not open USD file: {USD_FILE}")

    mesh_prims = [
        prim for prim in stage.Traverse()
        if prim.IsA(UsdGeom.Mesh)
    ]

    print(f"Found {len(mesh_prims)} mesh prims")

    for prim in mesh_prims:
        export_mesh(prim, stage, OUTPUT_DIR)

    print(f"Finished. STL files are in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
