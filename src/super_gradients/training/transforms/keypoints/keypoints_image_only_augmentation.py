from copy import deepcopy

from super_gradients.common.registry.registry import register_transform
from .abstract_keypoints_transform import AbstractKeypointTransform


@register_transform()
class KeypointsImageOnlyAugmentation(AbstractKeypointTransform):
    """Appearance changes before framing, without altering annotation eligibility.

    Padding is introduced afterwards, so noise/compression cannot brighten it.
    Geometry is explicitly rejected: this wrapper does not transform coordinates.
    """

    def __init__(self, transforms):
        super().__init__()
        import albumentations as A
        from super_gradients.common.factories.transforms_factory import AlbumentationsTransformsFactory

        self.transform = AlbumentationsTransformsFactory().get({"Compose": {"transforms": deepcopy(transforms)}})
        if not all(isinstance(t, A.ImageOnlyTransform) for t in self.transform.transforms):
            raise ValueError("Only image-only appearance transforms are allowed")

    def apply_to_sample(self, sample):
        sample.image = self.transform(image=sample.image)["image"]
        return sample

    def get_equivalent_preprocessing(self):
        raise RuntimeError("Random appearance augmentation has no equivalent inference preprocessing")
