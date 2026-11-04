# E-ink investigation notes (handoff)

Working notes from a cloud session, for continuing locally. Not meant for an upstream PR: drop this
file before opening one against mpreg-ca/webgpuviewer.

Device: BOOX Go 10.3 (Gen 2), Android 15. Mihon preview r8217 (mihonapp/mihon `cc465fe`,
webgpuviewer v54) works on it with "Use high quality renderer" on.

## 1. The ripple

With the high quality renderer on, each page turn is followed by a "ripple" spreading from the centre
to the edges, block by block, a couple of times a second. Faint in the panel's fast refresh mode,
severe in quality mode.

Cause: `TileRenderer` (renderer/TileRenderer.kt) draws the page with the cheap `RenderPage.renderFast`
first, then a background worker (`schedule()`) generates high-quality tiles centre-out
(`nextRequest()`), in batches of about 4 ms (`BATCH_TARGET_NS`, max 8 tiles). After every batch that
includes a visible tile it calls `invalidate()`, so every batch is a new frame, and on e-ink every
frame is a separate panel refresh.

Reported upstream draft (not yet filed): see the conversation; Mihon issue form fields were prepared.

## 2. Fix on this branch: `progressive` / `progressiveTiles`

`TileRenderer.progressive` (public as `ImageViewerState.progressiveTiles`, default true = unchanged).
When false:

- `drawCore()` keeps a grid's tiles off screen (returns not covered, so the page draws the fast way)
  until every visible tile is cut and none is owed a filtered re-cut; then `PageTiles.revealed` is set
  and the tiles show in one frame. `revealed` resets when the grid is wiped (`releaseTiles`,
  `destroyAll`).
- A revealed grid keeps its tile size instead of re-cutting to a new `preferredTileSize` (that would
  hide it again).
- The worker only calls `invalidate()` once a grid that got a visible tile has nothing visible left
  in `pending` (`visiblePending()`), instead of after every batch.

Mihon side: hlibkoval/mihon PR #1, branch `eink-prevent-ripple` - switch "Prevent e-ink ripple"
under Settings > Reader > E-Ink, above "Flash on page change"; sets
`viewerState.progressiveTiles = !showTilesAtOnce`.

Tested on the BOOX (patched v54, see 4): no ripple. One final refresh remains when the tiles appear,
and the user sees no quality difference after it.

## 3. Likely better approach: skip the quality pass

The fast render is already better than the legacy viewer, and on this ~300 dpi screen the tiles add
nothing visible. `ImagePage.ImageSingle.highQuality = false` skips `TileRenderer` entirely (no tiles,
no prewarm, no final refresh) and draws exactly the fast render. Mihon-only change in
`WebGpuViewer.kt`: it sets `it.highQuality = !animated` when a preview becomes the final image, and
leaves the default (true) on the new-page path (`ImagePage.ImageSingle(firstImage)`). Make both
`!animated && !preventRipple`, and re-apply to cached pages when the setting flips. Downside: zoomed-in
pages stay on bilinear instead of Catmull-Rom. Not implemented yet - user was choosing between this and
the progressive fix.

## 4. Crash: newer Dawn fails on the BOOX

Symptom: reader crashes on open with `UninitializedPropertyAccessException: lateinit property device
has not been initialized` from `TileRenderer.<clinit>`. Real cause is earlier: `WebGpuRenderer`'s
companion `init` failed and set `initError`; the actual error is only in logcat
(`WebGpuRenderer: Failed to initialize WebGPU`). Not captured yet - get it with
`adb logcat -s WebGpuRenderer`.

Why: the library embeds `androidx.webgpu:webgpu:1.0.0-dev06` from
https://github.com/mpreg-ca/androidx-webgpu-repo. That artifact was replaced in place on 2026-10-27
("Update Dawn to b11ffea…", commit `857991d`). Official v54 (2026-10-26) has the older build
(`libwebgpu_c_bundled.so` arm64 = 9,721,032 bytes); v56 and current master get the new one
(9,781,000 bytes). Builds with the new Dawn crash on the BOOX; patched v54 built against the old
artifact (repo commit `765b859`) works. v56 also adds a 64-byte immediates check in
`WebGpuRenderer` init - may matter too, unconfirmed.

Consequence: Mihon main moved to v56 (#4226), so upcoming previews will likely crash on this device.
Worth an issue once the logcat error is known. Mihon also has no fallback when `initError` is set.

To build against the old Dawn: clone androidx-webgpu-repo, `git checkout 765b859`, and point the
`maven { url = ... }` for it in `settings.gradle.kts` at `file:///path/to/androidx-webgpu-repo`.

## 5. Prewarm doesn't help after the first turn (code reading, unconfirmed on device)

`prewarm()` refuses when `total + missing * tileBytes > maxTileBytes`. Budget is `cacheScreens = 1.5`
screens at 256 px, ring included: about 180 tiles (45 MB) at 1860x2480. `RETAIN_MARGIN = 2` keeps the
previous page's grid, so previous + current + next (about 3 x 80 tiles) exceeds it and prewarm returns
early. It inserts the empty next grid first, so the next frame's retain trim would drop the previous
page and let it fit - but no further frame is requested once the current page is fully tiled, and
`evict()` only runs during tile generation. Confirm with `adb logcat -s TileRenderer` ("Pre-warming
next page tiles" should then appear only on the first turn). Fix ideas: free the previous grid (or
evict) before refusing; or invalidate once after inserting; or a larger budget.

## 6. Legacy viewer softness (for reference)

`ReaderPageImageView` sets `setMinimumTileDpi(180)`; SSIV decodes at the lowest power-of-two sample
size that still gives 180 dpi, i.e. 60% of this screen's pixels, so source pages wider than about
2,800 px decode at half size. No mipmaps (`setHasMipMap` never set) either.

## Branches

- hlibkoval/webgpuviewer `eink-progressive-tiles`: fix on master (`9f3ed58`) + these notes.
- hlibkoval/webgpuviewer `eink-progressive-tiles-v54`: same fix on tag 54 (`74ff360`) - the variant
  tested on the device (needs the old Dawn, see 4).
- hlibkoval/mihon `eink-prevent-ripple`: Mihon switch on main (needs a library with
  `progressiveTiles`; PR #1).
- hlibkoval/mihon `eink-prevent-ripple-r8217`: same commit on `cc465fe` (preview r8217, v54) - the
  Mihon side of the tested build.
