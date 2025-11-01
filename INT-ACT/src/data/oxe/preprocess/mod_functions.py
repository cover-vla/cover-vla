from abc import ABC, abstractmethod
from typing import ClassVar

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_graphics.geometry.transformation as tfg

import src.data.dlimp as dl

# from src.utils.geometry import euler2quat


class TfdsModFunction(ABC):
    @classmethod
    @abstractmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        """
        Modifies the data builder feature dict to reflect feature changes of ModFunction.
        """
        ...

    @classmethod
    @abstractmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        """
        Perform arbitrary modifications on the dataset that comply with the modified feature definition.
        """
        ...


def mod_obs_features(features, obs_feature_mod_function):
    """Utility function to only modify keys in observation dict."""
    return tfds.features.FeaturesDict(
        {
            "steps": tfds.features.Dataset(
                {
                    "observation": tfds.features.FeaturesDict(
                        {
                            key: obs_feature_mod_function(
                                key, features["steps"]["observation"][key]
                            )
                            for key in features["steps"]["observation"].keys()
                        }
                    ),
                    **{
                        key: features["steps"][key]
                        for key in features["steps"].keys()
                        if key not in ("observation",)
                    },
                }
            ),
            **{key: features[key] for key in features.keys() if key not in ("steps",)},
        }
    )


class ResizeAndJpegEncode(TfdsModFunction):
    MAX_RES: int = 224

    @classmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        def downsize_and_jpeg(key, feat):
            """Downsizes image features, encodes as jpeg."""
            if (
                len(feat.shape) >= 2 and feat.shape[0] >= 64 and feat.shape[1] >= 64
            ):  # is image / depth feature
                should_jpeg_encode = (
                    isinstance(feat, tfds.features.Image) and "depth" not in key
                )
                if len(feat.shape) > 2:
                    new_shape = (
                        ResizeAndJpegEncode.MAX_RES,
                        ResizeAndJpegEncode.MAX_RES,
                        feat.shape[2],
                    )
                else:
                    new_shape = (
                        ResizeAndJpegEncode.MAX_RES,
                        ResizeAndJpegEncode.MAX_RES,
                    )

                if isinstance(feat, tfds.features.Image):
                    return tfds.features.Image(
                        shape=new_shape,
                        dtype=feat.dtype,
                        encoding_format="jpeg" if should_jpeg_encode else "png",
                        doc=feat.doc,
                    )
                else:
                    return tfds.features.Tensor(
                        shape=new_shape,
                        dtype=feat.dtype,
                        doc=feat.doc,
                    )

            return feat

        return mod_obs_features(features, downsize_and_jpeg)

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        def resize_image_fn(step):
            # resize images
            for key in step["observation"]:
                if len(step["observation"][key].shape) >= 2 and (
                    step["observation"][key].shape[0] >= 64
                    or step["observation"][key].shape[1] >= 64
                ):
                    size = (ResizeAndJpegEncode.MAX_RES, ResizeAndJpegEncode.MAX_RES)
                    if "depth" in key:
                        step["observation"][key] = tf.cast(
                            dl.utils.resize_depth_image(
                                tf.cast(step["observation"][key], tf.float32), size
                            ),
                            step["observation"][key].dtype,
                        )
                    else:
                        step["observation"][key] = tf.cast(
                            dl.utils.resize_image(step["observation"][key], size),
                            tf.uint8,
                        )
            return step

        def episode_map_fn(episode):
            episode["steps"] = episode["steps"].map(resize_image_fn)
            return episode

        return ds.map(episode_map_fn)

class ConvertProprioToQuaternion(TfdsModFunction):
    @classmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        def convert_state_feature(key, feat):
            # Only modify the 'state' feature if it is a Tensor of shape (7,)
            if key == "state" and hasattr(feat, "shape") and len(feat.shape) == 1 and feat.shape[0] == 7:
                return tfds.features.Tensor(
                    shape=(8,),
                    dtype=feat.dtype,
                    doc=feat.doc,
                )
            return feat

        return mod_obs_features(features, convert_state_feature)

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        def convert_state(step):
            if "state" in step["observation"]:
                state = step["observation"]["state"]
                if state.shape[0] == 7:
                    proprio = state[3:6]
                    quat_xyzw = tfg.quaternion.from_euler(angles=proprio)
                    step["observation"]["state"] = tf.concat([state[:3], quat_xyzw, state[6:]],
                                                            axis=0)
            return step

        def episode_map_fn(episode):
            episode["steps"] = episode["steps"].map(convert_state)
            return episode

        return ds.map(episode_map_fn)

class ConvertProprioToEuler(TfdsModFunction):
    # this only works for fractal since the keys are hardcoded
    @classmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        def convert_state_feature(key, feat):
            # fractal dataset proprio is of shape (7,) without gripper
            if key == "base_pose_tool_reached" and hasattr(feat, "shape") and len(feat.shape) == 1 and feat.shape[0] == 7:
                return tfds.features.Tensor(
                    shape=(6,),
                    dtype=feat.dtype,
                    doc=feat.doc,
                )
            return feat

        return mod_obs_features(features, convert_state_feature)

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        def convert_state(step):

            state = step["observation"]["base_pose_tool_reached"]

            proprio = state[3:7] # xyzw quaternion
            euler = tfg.euler.from_quaternion(quaternions=proprio)
            step["observation"]["base_pose_tool_reached"] = tf.concat([state[:3], euler],
                                                    axis=0)
            return step

        def episode_map_fn(episode):
            episode["steps"] = episode["steps"].map(convert_state)
            return episode

        return ds.map(episode_map_fn)

class FilterSuccess(TfdsModFunction):
    @classmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        return features  # no feature changes

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds.filter(lambda e: e["success"])

class FlipImgChannels(TfdsModFunction):
    FLIP_KEYS: ClassVar[list[str]] = ["image"]

    @classmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        return features  # no feature changes

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        def flip(step):
            for key in cls.FLIP_KEYS:
                if key in step["observation"]:
                    step["observation"][key] = step["observation"][key][..., ::-1]
            return step

        def episode_map_fn(episode):
            episode["steps"] = episode["steps"].map(flip)
            return episode

        return ds.map(episode_map_fn)


class FlipWristImgChannels(FlipImgChannels):
    FLIP_KEYS: ClassVar[list[str]] = ["wrist_image", "hand_image"]


TFDS_MOD_FUNCTIONS = {
    "resize_and_jpeg_encode": ResizeAndJpegEncode,
    "filter_success": FilterSuccess,
    "flip_image_channels": FlipImgChannels,
    "flip_wrist_image_channels": FlipWristImgChannels,
    "convert_proprio_to_quaternion": ConvertProprioToQuaternion,
    "convert_proprio_to_euler": ConvertProprioToEuler,
}
