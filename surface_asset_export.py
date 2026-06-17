import os
from pathlib import Path

from plyfile import PlyData
from volumetric_bodyfit.config.runtime import inner_points_source_dir


def convert_ply_mesh_to_obj(source_mesh: Path, target_mesh: Path) -> None:
    ply_data = PlyData.read(source_mesh)
    target_mesh.parent.mkdir(parents=True, exist_ok=True)

    with target_mesh.open("w", encoding="utf-8") as stream:
        if "vertex" in ply_data:
            for vertex in ply_data["vertex"]:
                stream.write(f"v {vertex['x']} {vertex['y']} {vertex['z']}\n")

        if "face" in ply_data:
            for face in ply_data["face"]:
                indices = " ".join(str(int(vertex_id) + 1) for vertex_id in face["vertex_indices"])
                stream.write(f"f {indices}\n")


def export_inner_point_meshes(source_root: Path, output_root: Path) -> None:
    if not source_root.exists():
        raise FileNotFoundError(f"Input folder does not exist: {source_root}")

    for subject_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        ply_file = subject_dir / f"pred_inner_points_{subject_dir.name}.ply"
        if not ply_file.exists():
            print(f"Missing input mesh: {ply_file}")
            continue

        obj_file = output_root / subject_dir.name / f"{subject_dir.name}.obj"
        print(f"Converting {ply_file} -> {obj_file}")
        convert_ply_mesh_to_obj(ply_file, obj_file)


if __name__ == "__main__":
    source_directory = Path(os.environ.get("BODYFIT_INNER_POINT_SOURCE", inner_points_source_dir))
    export_inner_point_meshes(source_directory, Path("inner_cape_smplx"))

