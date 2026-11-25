import hashlib
import os
import re
import tempfile
from pathlib import Path

import xacro
import yaml
from launch.actions import SetEnvironmentVariable


def flatten_dict(src, parent_key="", out=None):
    if out is None:
        out = {}

    for key, value in src.items():
        full_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            flatten_dict(value, full_key, out)
        else:
            out[full_key] = value
    return out


def load_robot_params_as_ros_dict(global_parameters):
    with open(global_parameters, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp) or {}
    return flatten_dict(config)


def resolve_default_config_root():
    product_root = "/robros/configs"
    if os.path.exists(product_root):
        return product_root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(repo_root, "configs")


def _resolve_dev_repo_root():
    env_root = os.environ.get("IGRIS_C_REPO_ROOT", "").strip()
    if env_root and os.path.isdir(os.path.join(env_root, "configs", "robot")):
        return env_root

    here = os.path.abspath(os.path.dirname(__file__))
    cur = here
    while True:
        if os.path.isdir(os.path.join(cur, "configs", "robot")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    install_marker = f"{os.sep}install{os.sep}"
    if install_marker in here:
        ws_root = here.split(install_marker, 1)[0]
        candidate = os.path.join(ws_root, "src", "igris_c")
        if os.path.isdir(os.path.join(candidate, "configs", "robot")):
            return candidate

    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def cyclonedds_env_actions(ros_params):
    """SetEnvironmentVariable actions for cyclonedds settings sourced from
    robot.yaml: CYCLONEDDS_URI and ROS_DOMAIN_ID. The domain id is mirrored
    onto ROS_DOMAIN_ID so rclcpp nodes spawned by this launch live on the
    same DDS partition as the SDK clients reading from robot.yaml.
    """
    actions = []
    cyclonedds_uri = ros_params.get("igris_c.network.cyclonedds.uri_file", "")
    if cyclonedds_uri:
        actions.append(SetEnvironmentVariable(name="CYCLONEDDS_URI", value=cyclonedds_uri))
    domain_id = ros_params.get("igris_c.network.cyclonedds.domain_id")
    if domain_id is None:
        raise RuntimeError(
            "Missing required robot.yaml key: igris_c.network.cyclonedds.domain_id. "
            "Add it under network.cyclonedds.domain_id so launch can mirror the value onto ROS_DOMAIN_ID."
        )
    actions.append(SetEnvironmentVariable(name="ROS_DOMAIN_ID", value=str(domain_id)))
    return actions


def resolve_robot_config_path(config_file, launch_file):
    config_file = (config_file or "").strip()
    repo_root = _resolve_dev_repo_root()
    dev_config_hint = os.path.join(repo_root, "configs", "robot", "dev_robot.yaml")

    if not config_file:
        raise RuntimeError(
            "Missing required launch argument: config_file\n"
            f"Hint (installed): ros2 launch igris_c_controller {launch_file} "
            "config_file:=/robros/configs/robot.yaml\n"
            f"Hint (dev): ros2 launch igris_c_controller {launch_file} "
            f"config_file:={dev_config_hint}"
        )

    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"Config file not found: {config_file}\n"
            "Hint (installed): /robros/configs/robot.yaml\n"
            f"Hint (dev): {dev_config_hint}"
        )
    return config_file


def _validate_end_effector_type(end_effector_type):
    normalized = str(end_effector_type).strip()
    if normalized not in {"dummy", "none", "hand", "magnet", "1dof"}:
        raise RuntimeError(
            f"Unsupported igris_c.end_effector.type: {end_effector_type!r}. "
            "Expected one of: dummy, none, hand, magnet, 1dof."
        )
    return normalized


def _resolve_description_root(ros_params):
    description_root = str(ros_params.get("igris_c.description.config_root_dir", "")).strip()
    if not description_root:
        raise RuntimeError("Missing igris_c.description.config_root_dir")
    root = Path(description_root).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"igris_c.description.config_root_dir is not a directory: {root}")
    return root


def _resolve_description_path(description_root, path_value, label):
    path_value = str(path_value or "").strip()
    if not path_value:
        raise RuntimeError(f"Missing {label}")

    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = description_root / candidate
    return candidate.resolve()


def generate_urdf_xml(model_path, mappings):
    model_path = Path(model_path).resolve()
    model_path_str = str(model_path)
    if model_path_str.endswith(".urdf"):
        return model_path.read_text(encoding="utf-8")
    if not model_path_str.endswith(".urdf.xacro"):
        raise RuntimeError(
            f"Unsupported igris_c.description.urdf.robot_model_path: {model_path}. "
            "Expected .urdf or .urdf.xacro."
        )
    return xacro.process_file(model_path_str, mappings=mappings).toxml()


def _absolutize_mujoco_compiler_attr(xml_text, xacro_dir, attr_name):
    pattern = re.compile(rf'({attr_name}=")([^"]+)(")')
    match = pattern.search(xml_text)
    if not match:
        return xml_text

    attr_path = Path(match.group(2))
    if attr_path.is_absolute():
        return xml_text

    absolute_path = str((xacro_dir / attr_path).resolve())
    return xml_text[: match.start(2)] + absolute_path + xml_text[match.end(2) :]


def generate_mujoco_xml_path(model_path, mappings):
    model_path = Path(model_path).resolve()
    model_path_str = str(model_path)
    if model_path_str.endswith(".xml"):
        return model_path_str
    if not model_path_str.endswith(".xml.xacro"):
        raise RuntimeError(
            f"Unsupported igris_c.description.mujoco.robot_model_path: {model_path}. "
            "Expected .xml or .xml.xacro."
        )

    xml_text = xacro.process_file(model_path_str, mappings=mappings).toxml()
    xml_text = _absolutize_mujoco_compiler_attr(xml_text, model_path.parent, "meshdir")
    xml_text = _absolutize_mujoco_compiler_attr(xml_text, model_path.parent, "texturedir")

    output_dir = Path(tempfile.gettempdir()) / "igris_c_generated_mujoco"
    output_dir.mkdir(parents=True, exist_ok=True)

    digest_input = f"{model_path}|{sorted(mappings.items())}"
    digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:16]
    output_path = output_dir / f"igris_c_v2_{digest}.xml"
    output_path.write_text(xml_text, encoding="utf-8")
    return str(output_path)


def prepare_model_params(ros_params):
    prepared = dict(ros_params)
    description_root = _resolve_description_root(prepared)

    xacro_end_effector = _validate_end_effector_type(
        prepared.get("igris_c.end_effector.type", "none")
    )

    urdf_model_path = _resolve_description_path(
        description_root,
        prepared.get("igris_c.description.urdf.robot_model_path", ""),
        "igris_c.description.urdf.robot_model_path",
    )
    prepared["igris_c.description.urdf.robot_model_path"] = str(urdf_model_path)
    prepared["igris_c.description.urdf.robot_model_xml"] = generate_urdf_xml(
        urdf_model_path,
        {
            "base_type": "pelvis",
            "parallel": "false",
            "end_effector": xacro_end_effector,
        },
    )

    parallel_links_path = _resolve_description_path(
        description_root,
        prepared.get("igris_c.description.urdf.parallel_links_path", ""),
        "igris_c.description.urdf.parallel_links_path",
    )
    prepared["igris_c.description.urdf.parallel_links_path"] = str(parallel_links_path)

    mujoco_model_path = _resolve_description_path(
        description_root,
        prepared.get("igris_c.description.mujoco.robot_model_path", ""),
        "igris_c.description.mujoco.robot_model_path",
    )
    prepared["igris_c.description.mujoco.robot_model_path"] = generate_mujoco_xml_path(
        mujoco_model_path,
        {
            "base_type": str(
                prepared.get("igris_c.description.mujoco.base_type", "pelvis")
            ).strip(),
            "parallel": str(prepared.get("igris_c.description.mujoco.parallel", "true"))
            .strip()
            .lower(),
            "end_effector": xacro_end_effector,
            "fixed": str(prepared.get("igris_c.description.mujoco.fixed", "true")).strip().lower(),
        },
    )
    return prepared
