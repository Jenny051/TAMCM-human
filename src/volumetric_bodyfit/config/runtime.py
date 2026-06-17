import os
from pathlib import Path


def _env_path(name: str, fallback: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else fallback.resolve()


def _folder(path: Path) -> str:
    return str(path) + os.sep


package_dir = Path(__file__).resolve().parent
project_home_path = _env_path("BODYFIT_PROJECT_HOME", package_dir.parents[1])
artifact_root_path = _env_path("BODYFIT_ARTIFACT_ROOT", project_home_path)

project_home = str(project_home_path)
support_data_dir = artifact_root_path / "support_data"

faust_scan_dir = _folder(_env_path("BODYFIT_FAUST_SCAN_DIR", project_home_path / "datasets" / "faust_scans"))
faust_registration_dir = _folder(
    _env_path("BODYFIT_FAUST_REGISTRATION_DIR", project_home_path / "datasets" / "faust_registrations")
)

challenge_pair_file = str(
    project_home_path / "src" / "volumetric_bodyfit" / "resources" / "faust_pairs.txt"
)
body_vertex_group_file = str(project_home_path / "src" / "volumetric_bodyfit" / "resources" / "body_vertex_groups.json")

neutral_body_model_path = str(support_data_dir / "body_models" / "smplh" / "neutral" / "model.npz")
registration_output_dir = _folder(_env_path("BODYFIT_OUTPUT_DIR", project_home_path / "output"))
checkpoint_store = _folder(_env_path("BODYFIT_CHECKPOINT_DIR", artifact_root_path / "storage"))
model_profile_store = _folder(_env_path("BODYFIT_MODEL_PROFILE_DIR", project_home_path / "model_profiles"))

demo_scan_dir = _folder(_env_path("BODYFIT_DEMO_SCAN_DIR", project_home_path / "demo"))
rotated_demo_scan_dir = _folder(_env_path("BODYFIT_ROTATED_DEMO_SCAN_DIR", project_home_path / "demo_guess_rot"))
processed_training_data_dir = _folder(_env_path("BODYFIT_PROCESSED_DATA_DIR", project_home_path / "training_tmp" / "processed_data"))

cape_eval_scan_dir = _folder(_env_path("BODYFIT_CAPE_EVAL_DIR", project_home_path / "datafolder" / "CAPE" / "eval"))
cape_raw_scan_dir = _folder(_env_path("BODYFIT_CAPE_RAW_DIR", project_home_path / "datafolder" / "CAPE" / "raw_eval"))
cape_smpl_info_dir = _folder(_env_path("BODYFIT_CAPE_SMPL_DIR", project_home_path / "datafolder" / "CAPE" / "smpl"))

dress_eval_scan_dir = _folder(_env_path("BODYFIT_DRESS_EVAL_DIR", project_home_path / "datafolder" / "4D-DRESS" / "eval"))
dress_smpl_info_dir = _folder(_env_path("BODYFIT_DRESS_SMPL_DIR", project_home_path / "datafolder" / "4D-DRESS" / "smplh"))

behave_laplacian_path = str(_env_path("BODYFIT_BEHAVE_LAPLACIAN", project_home_path / "support_data" / "behave_laplacian.pkl"))
inner_points_source_dir = str(_env_path("BODYFIT_INNER_POINT_SOURCE", project_home_path / "inner_points_source"))

