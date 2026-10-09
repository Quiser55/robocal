"""Shared launch helpers for the AR4 Gazebo launch files.

Imported by both `gazebo.launch.py` and `calibration_sim.launch.py` via a
`sys.path` insert of their own directory - launch files are installed into
`share/annin_ar4_gazebo/launch/` side by side, so this works identically from
a source checkout and from an installed workspace.
"""
import os
import tempfile

from launch.substitution import Substitution


def gz_resource_path_value() -> str:
    """Value for ``GZ_SIM_RESOURCE_PATH``, covering every colcon prefix.

    sdformat's URDF parser rewrites ``package://X/...`` mesh URIs into
    ``model://X/...``, and gz-common then resolves those *only* by searching
    ``GZ_SIM_RESOURCE_PATH``. Nothing in ROS 2 or colcon sets that variable, so
    without this the AR4's visual **and collision** meshes silently fail to
    load and the model appears empty in Gazebo.
    """
    prefixes = [p for p in os.environ.get('AMENT_PREFIX_PATH', '').split(os.pathsep) if p]
    entries = [os.path.join(prefix, 'share') for prefix in prefixes]

    existing = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    if existing:
        entries.extend(e for e in existing.split(os.pathsep) if e)

    seen = set()
    return os.pathsep.join(e for e in entries if not (e in seen or seen.add(e)))


class ControllerConfigSubstitution(Substitution):
    """Substitution that fills out tf_prefix in controllers.yaml."""

    def __init__(self, file_path: Substitution, tf_prefix: Substitution):
        super().__init__()
        self._file_path = file_path
        self._tf_prefix = tf_prefix

    def perform(self, context):
        # Evaluate the file path and namespace substitutions
        file_path_val = self._file_path.perform(context)
        tf_prefix_val = self._tf_prefix.perform(context)

        with open(file_path_val, "r") as f:
            content = f.read()

        content = content.replace('$(var tf_prefix)', tf_prefix_val)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
        temp_file.write(content.encode("utf-8"))
        temp_file.close()
        return temp_file.name


class PerturbedUrdfSubstitution(Substitution):
    """Bake a known kinematic error into the URDF that Gazebo spawns, and
    return the path it was written to.

    This is what gives the simulation ground truth: Gazebo gets the
    *perturbed* robot while `robot_state_publisher` (and therefore MoveIt,
    ros2_control and every calibration node) keeps the *nominal* description.
    WP4 then has a real error to identify whose true value is known exactly.

    Consequently the spawner must use ``-file`` with this path, never
    ``-topic robot_description`` - that topic is deliberately the nominal model.
    """

    def __init__(self, urdf: Substitution, perturbation_file: Substitution,
                 out_path: Substitution, tf_prefix: Substitution):
        super().__init__()
        self._urdf = urdf
        self._perturbation_file = perturbation_file
        self._out_path = out_path
        self._tf_prefix = tf_prefix

    def perform(self, context):
        from annin_ar4_calibration.core import paths, perturbation

        urdf_xml = self._urdf.perform(context)
        corrections = perturbation.load(self._perturbation_file.perform(context))
        out_path = self._out_path.perform(context) or paths.default_perturbed_urdf_file()

        perturbed = perturbation.apply_to_urdf(
            urdf_xml, corrections, tf_prefix=self._tf_prefix.perform(context))
        with open(out_path, 'w') as f:
            f.write(perturbed)
        return out_path
