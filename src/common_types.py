from typing import TypeAlias
import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]

PointCloud3N: TypeAlias = FloatArray
PointsN3: TypeAlias = FloatArray
Centers3K: TypeAlias = FloatArray
NormalsN3: TypeAlias = FloatArray

Alpha: TypeAlias = FloatArray
PrecisionVector: TypeAlias = FloatArray
PrecisionMatrices: TypeAlias = FloatArray
Priors: TypeAlias = FloatArray

Rotation: TypeAlias = FloatArray
Translation: TypeAlias = FloatArray
Transform: TypeAlias = tuple[Rotation, Translation]

ViewTransforms: TypeAlias = tuple[list[Rotation], list[Translation]]
EMTransformState = tuple[ViewTransforms]

def as_translation(x: FloatArray) -> Translation:
    return np.asarray(x, dtype=np.float64).reshape(3)


def as_rotation(x: FloatArray) -> Rotation:
    return np.asarray(x, dtype=np.float64).reshape(3, 3)
