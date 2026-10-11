# Chess augmentation order

The chess pose recipe uses:

1. Brightness, contrast, and HSV augmentation.
2. Board-aware random framing into 640 x 480 with black padding.
3. Combined rotation, shear, and optional perspective distortion.
4. Blur, noise, JPEG compression, and random masking at the final resolution.
5. Normalization and removal of empty/small instances.

Framing before geometry prevents rotation from clipping a board against the original
portrait or panoramic image boundaries before the eventual 4:3 canvas is available.
It does not guarantee that every rotation fits in 640 x 480.

`KeypointsBoardSafeGeometry` samples rotation up to 90 degrees in either direction,
shear up to 5 degrees on each axis, and inward corner perspective distortion up to
15% with probability 0.5. It composes the matrices and checks the final coordinates
before rendering one image warp. There is no extra random scaling or translation.

Each board must retain at least 75% of its initially visible landmarks, rounded up,
and at least four when four were present. Thus a complete nine-point board may lose
two landmarks; an already partial board is assessed against its visible landmarks.
Pre-existing invisible points are never made visible. The check is per board, not
an aggregate across multiple boards. Samples with no visible board are unconstrained.

Candidates that exceed the clipping limit are resampled, up to 20 attempts. If all
fail, geometry is the identity. This intentionally makes the accepted angle and
perspective distributions depend on the framing; it does not shrink the board to
fit a rejected angle. The limits and attempt count are recipe parameters.

Validation retains deterministic board-aware framing and uses black padding. It has
no random geometry or appearance augmentation. Changing this recipe affects future
runs; it does not alter an existing run's resolved configuration.

The crop retains its existing size policy: it never zooms in beyond the largest
inscribed 4:3 window. Naturally small boards and already incomplete source boards
remain possible. Pixel-sized blur/noise/masking now acts at 640 x 480, so these effects
can be stronger than when they were applied to a large photograph before downsampling.
