# Chess augmentation order

The chess pose recipe uses:

1. Brightness, contrast, HSV, blur, noise, and JPEG compression.
2. Board-aware random framing into 640 x 480 with black padding.
3. Combined rotation, shear, and optional perspective distortion.
4. Normalization and removal of empty/small instances.

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
Missing/unannotated points are never made valid. The check is per board, not
an aggregate across multiple boards. Samples with no visible board are unconstrained.

Candidates that exceed the clipping limit are resampled, up to 20 attempts. If all
fail, geometry is the identity. This intentionally makes the accepted angle and
perspective distributions depend on the framing; it does not shrink the board to
fit a rejected angle. The limits and attempt count are recipe parameters.

Validation uses `KeypointsLetterbox`: resize the entire image proportionally to fit
640 x 480, then center with black padding on only one axis (or none for a 4:3 image).
There is no board-dependent crop or extra margin. Opposite padding widths differ by
at most one pixel. Dataset and inference share the same image processors, including
rounding and inverse coordinate mapping.

Both new dataset recipes enable `preserve_out_of_frame_board_annotations`. Known
finite board annotations retain their coordinates through initial source clipping
and training framing; they are eligible if they enter the final canvas, including
padding. Missing annotations (v=0) remain missing; points outside the final canvas
are excluded. Piece annotations still require an in-source target. This flag is
opt-in so saved historical recipes keep their previous annotation policy.

Every configured image padding/fill value is zero. Random square masking is removed.
Appearance effects run before framing and geometry so they cannot brighten padding.
The image-only wrapper leaves annotations untouched and rejects geometric transforms.
Changing this recipe affects future runs; it does not alter an existing run's
resolved configuration.

The crop retains its existing size policy: it never zooms in beyond the largest
inscribed 4:3 window. Naturally small boards and already incomplete source boards
remain possible. Pixel-sized blur/noise acts at source resolution, before downsampling;
its apparent strength therefore depends on source resolution.
