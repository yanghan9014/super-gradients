# Chess augmentation order

The chess pose recipe uses:

1. Board-aware random framing of a 4:3 window. The full-resolution photo is shrunk once,
   by area averaging, to the output scale; the window itself is cut in step 3.
2. Brightness, contrast and HSV (blur, noise and JPEG compression are currently disabled).
3. Combined rotation, shear, and optional perspective distortion, warped together with the
   framing cut in one cubic resample to 640 x 480 with black padding.
4. Normalization and removal of empty/small instances.

So each training image is resampled twice in total: one area shrink and one near-1x cubic
warp. Previously a half-size JPEG decode, an area shrink and two bilinear warps made
training images about 2.7x softer (Laplacian variance) than this.

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

Validation uses `KeypointsBoardFocusLetterbox`: resize the entire image proportionally
(area averaging) to fit 640 x 480, then center with black padding on only one axis (or
none for a 4:3 image). Opposite padding widths differ by at most one pixel.

The exception is a board the app would refuse as too small: its a1-a8-h8-h1 outline
covering less than 0.12 of the letterboxed frame (`AppConfig.minBoardFrameFraction`).
Then the photo is first cropped to the board plus two squares on every side, measured
through the annotation homography, widened to 4:3 around the board and slid back inside
the photo. No padding is added for margin the photo lacks; an axis the crop cannot fit
keeps the whole photo. Pieces outside the crop are dropped. The crop follows only from
the annotations, so validation frames are identical on every run. Inference has no
annotations and uses the plain whole-image letterbox. The SightCheck evaluator imports
the same window function.

Both new dataset recipes enable `preserve_out_of_frame_board_annotations`. Known
finite board annotations retain their coordinates through initial source clipping
and training framing; they are eligible if they enter the final canvas, including
padding. Missing annotations (v=0) remain missing; points outside the final canvas
are excluded. Piece annotations still require an in-source target. This flag is
opt-in so saved historical recipes keep their previous annotation policy.

Every configured image padding/fill value is zero. Random square masking is removed.
Appearance effects run before the geometry warp adds padding, so they cannot brighten it.
The image-only wrapper leaves annotations untouched and rejects geometric transforms.
Changing this recipe affects future runs; it does not alter an existing run's
resolved configuration.

The crop retains its existing size policy: it never zooms in beyond the largest
inscribed 4:3 window. Naturally small boards and already incomplete source boards
remain possible. Because of that policy, the shrink in step 1 discards no detail the
network sees; it also means appearance augmentation runs on a small image. Pixel-sized
blur/noise acts at the output scale, so its apparent strength does not depend on the source.

Piece boxes are rebuilt at load: each becomes the image extent of one board square
centred on the piece, using the homography from the board's annotated landmarks. The
annotated boxes are a fixed 64 source pixels, which ranged from 0.7x to 2.9x a square
across photo resolutions; boxes decide anchor assignment and the piece OKS scale.
