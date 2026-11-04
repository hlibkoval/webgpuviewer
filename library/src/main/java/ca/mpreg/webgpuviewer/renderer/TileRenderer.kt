package ca.mpreg.webgpuviewer.renderer

import android.graphics.Rect
import android.util.Log
import androidx.webgpu.BlendFactor
import androidx.webgpu.BlendOperation
import androidx.webgpu.BufferUsage
import androidx.webgpu.CompareFunction
import androidx.webgpu.Constants
import androidx.webgpu.FeatureName
import androidx.webgpu.FilterMode
import androidx.webgpu.GPUBindGroup
import androidx.webgpu.GPUBindGroupDescriptor
import androidx.webgpu.GPUBindGroupEntry
import androidx.webgpu.GPUBindGroupLayoutDescriptor
import androidx.webgpu.GPUBindGroupLayoutEntry
import androidx.webgpu.GPUBlendComponent
import androidx.webgpu.GPUBlendState
import androidx.webgpu.GPUBuffer
import androidx.webgpu.GPUBufferDescriptor
import androidx.webgpu.GPUColor
import androidx.webgpu.GPUColorTargetState
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUComputePassDescriptor
import androidx.webgpu.GPUDepthStencilState
import androidx.webgpu.GPUExtent3D
import androidx.webgpu.GPUFragmentState
import androidx.webgpu.GPUOrigin3D
import androidx.webgpu.GPUPassTimestampWrites
import androidx.webgpu.GPUPipelineLayoutDescriptor
import androidx.webgpu.GPUPrimitiveState
import androidx.webgpu.GPUQuerySet
import androidx.webgpu.GPUQuerySetDescriptor
import androidx.webgpu.GPURenderPassColorAttachment
import androidx.webgpu.GPURenderPassDescriptor
import androidx.webgpu.GPURenderPassEncoder
import androidx.webgpu.GPURenderPipeline
import androidx.webgpu.GPURenderPipelineDescriptor
import androidx.webgpu.GPUSamplerBindingLayout
import androidx.webgpu.GPUSamplerDescriptor
import androidx.webgpu.GPUShaderModuleDescriptor
import androidx.webgpu.GPUShaderSourceWGSL
import androidx.webgpu.GPUStencilFaceState
import androidx.webgpu.GPUTexelCopyTextureInfo
import androidx.webgpu.GPUTexture
import androidx.webgpu.GPUTextureBindingLayout
import androidx.webgpu.GPUTextureDescriptor
import androidx.webgpu.GPUTextureView
import androidx.webgpu.GPUVertexAttribute
import androidx.webgpu.GPUVertexBufferLayout
import androidx.webgpu.GPUVertexState
import androidx.webgpu.LoadOp
import androidx.webgpu.MapMode
import androidx.webgpu.OptionalBool
import androidx.webgpu.PrimitiveTopology
import androidx.webgpu.QueryType
import androidx.webgpu.SamplerBindingType
import androidx.webgpu.ShaderStage
import androidx.webgpu.StencilOperation
import androidx.webgpu.StoreOp
import androidx.webgpu.TextureFormat
import androidx.webgpu.TextureSampleType
import androidx.webgpu.TextureUsage
import androidx.webgpu.TextureViewDimension
import androidx.webgpu.VertexFormat
import androidx.webgpu.VertexStepMode
import ca.mpreg.webgpuviewer.filter.Lut3dTexture
import ca.mpreg.webgpuviewer.renderer.TileRenderer.Companion.BATCH_TARGET_NS
import ca.mpreg.webgpuviewer.renderer.TileRenderer.Companion.OFF_SCREEN_SCORE
import ca.mpreg.webgpuviewer.renderer.TileRenderer.Companion.SLAB_SIZE
import ca.mpreg.webgpuviewer.renderer.TileRenderer.Companion.STENCIL_BUFFER_COUNT
import ca.mpreg.webgpuviewer.renderer.TileRenderer.Companion.TILES_PER_BATCH_FALLBACK
import ca.mpreg.webgpuviewer.renderer.TileRenderer.Companion.TILE_SIZE
import ca.mpreg.webgpuviewer.renderer.TileRenderer.Companion.TILE_SIZES
import ca.mpreg.webgpuviewer.renderer.TileRenderer.Companion.TILE_SIZE_MARGIN
import ca.mpreg.webgpuviewer.renderer.TileRenderer.Companion.TILE_SIZE_SAMPLES
import ca.mpreg.webgpuviewer.viewer.ImagePage
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineExceptionHandler
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.joinAll
import kotlinx.coroutines.launch
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.round
import kotlin.math.sqrt
import kotlin.time.Duration.Companion.milliseconds

/**
 * Solve the (x, y) [RenderPage.render]/[RenderPage.renderFast] need so [image] lands centred at
 * screen position ([targetX], [targetY]) - inverts [Image.prepareForRender]'s placement math for
 * an arbitrary target.
 */
internal fun solveImagePlacement(
    targetX: Float,
    targetY: Float,
    imageScale: Float,
    image: Image,
    dstWidth: Float,
    dstHeight: Float
): Pair<Float, Float> {
    val x =
        (targetX - dstWidth / 2f) / (imageScale * dstWidth) - image.offsetX / dstWidth - WebGpuRenderer.offsetX
    val y =
        (targetY - dstHeight / 2f) / (imageScale * dstHeight) - image.offsetY / dstHeight - WebGpuRenderer.offsetY
    return x to y
}

/**
 * A cache of the filtered render, cut into square screen-resolution tiles - [TILE_SIZE] by
 * default, per grid via [preferredTileSize]. Every tile is a slot in one atlas texture
 * ([TileAtlas]), so a grid draws in a single instanced call.
 *
 * [RenderPage.render] is too expensive every frame; [RenderPage.renderFast] is cheap but
 * unfiltered. Each frame draws the fast path, then blits whatever filtered tiles already exist
 * on top, while a background worker fills in the rest a few at a time.
 *
 * One grid per whole [ImagePage.ImageSingle], not per image, so a spread's seam bakes into whichever tile
 * straddles it rather than meeting two independently-snapped layers.
 *
 * The same grid serves both viewers. The paged viewer has one page on screen at a time and
 * rounds its own anchor; the continuous viewer can have several pages' grids live at once, so it
 * rounds only the shared *camera* position and leaves each page's offset from it exact - keeps
 * adjacent pages' tiles pixel-aligned at their shared boundary (see that [draw] overload).
 *
 * Tiles live in content space: tile (tx, ty) holds the square tile-size pixels right/down of
 * the grid's anchor, so panning survives untouched. Changing scale (or a document position shift
 * in the continuous viewer) invalidates the whole grid, which regenerates once scale has held
 * stable for two frames.
 *
 * The whole grid snaps to the nearest screen pixel so every blit is an exact 1:1 texel copy. A
 * grid's bind group and instances are kept between frames, so a blit is one instanced draw, its
 * anchor and clip sent as immediates.
 *
 * Generation runs on the render thread but outside the render mutex, a few tiles at a time with a
 * suspend between batches so a queued frame always gets the thread back. Pending tiles are pulled
 * on-screen-first, centre-out.
 *
 * Everything here is touched only on the render thread. [cleanup] may be called from any thread
 * and posts its work there.
 */
internal class TileRenderer(private val invalidate: () -> Unit) {
    companion object {
        const val TILE_SIZE = 256

        private const val TILES_PER_BATCH_FALLBACK = 1
        private const val BATCH_TARGET_NS = 4_000_000.0
        private const val MAX_TILES_PER_BATCH = 8

        /** Image filter blocks one step makes before [blockCostNs] is known. */
        private const val BLOCKS_PER_STEP_FALLBACK = 2

        private const val MAX_BLOCKS_PER_STEP = 16

        /**
         * Grace window of extra pages (past "whichever is current") [draw] keeps a grid for, so
         * leaving a page doesn't force full regeneration on turning right back. Paged overload
         * only - the continuous viewer relies on [evict]'s shared LRU cap instead.
         */
        private const val RETAIN_MARGIN = 2

        private const val TAG = "TileRenderer"

        /** (tx, ty, atlas x, atlas y) as floats - see the blit shader's instance input. */
        private const val INSTANCE_BYTES = 16L

        /**
         * The atlas is carved into slabs, each subdivided into slots of one tile size. A slab is
         * the unit that changes hands when the preferred size moves, so it is large enough that a
         * class holds few of them and small enough that a half-used one wastes little.
         */
        private const val SLAB_SIZE = 512

        /** Smallest [preferredTileSize] worth having - below this the per-tile pass dominates. */
        private const val MIN_TILE_SIZE = 64

        /** The sizes [reconsiderTileSize] chooses between, smallest first. */
        private val TILE_SIZES = intArrayOf(MIN_TILE_SIZE, 128, 256, SLAB_SIZE)

        /** Measurements a size needs before it is allowed to win, or lose, a comparison. */
        private const val TILE_SIZE_SAMPLES = 4

        /** How much cheaper per pixel another size must look before the grids are re-cut. */
        private const val TILE_SIZE_MARGIN = 1.15

        /** FrameParams: snap, dst_size, clip, ts and the atlas's side, padded. */
        private const val FRAME_IMMEDIATE_BYTES = 48

        /** What one [TILE_SIZE] tile costs, and the range the derived cache is held to. */
        private const val TILE_BYTES = TILE_SIZE * TILE_SIZE * 4
        private const val MIN_CACHE_BYTES = 16 * 1024 * 1024
        private const val MAX_CACHE_BYTES = 64 * 1024 * 1024

        /**
         * Score threshold [nextRequest] uses to tell a genuinely on-screen tile request from one
         * outside a grid's wanted range (e.g. [prewarm]'s tiles, which leave that range empty).
         */
        private const val OFF_SCREEN_SCORE = 1e6f

        /**
         * Ring buffer size for [stencilViewFor] - matches a typical Android/Vulkan surface's
         * buffer count so a rotated stencil texture is never still in flight from a prior frame.
         */
        private const val STENCIL_BUFFER_COUNT = 3

        private val device get() = WebGpuRenderer.device

        private val timestampsSupported = device.hasFeature(FeatureName.TimestampQuery)

        // Nearest: tiles are blitted 1:1 at integer pixel positions, so this is an exact copy.
        private val blitSampler by lazy {
            device.createSampler(
                GPUSamplerDescriptor(
                    magFilter = FilterMode.Nearest,
                    minFilter = FilterMode.Nearest,
                )
            )
        }

        // Explicit (not auto-inferred) so it can be shared across
        // blitPipeline/blitPipelineStencilWrite - an implicit/auto pipeline layout is unique to the
        // pipeline it was inferred for, so a bind group made against one pipeline's auto layout is
        // invalid on the other, even though both pipelines share the exact same shader and
        // bindings. A grid's bind group is created once and reused across both pipelines (see
        // gridBindGroup), so it must be built against this.
        private val blitBindGroupLayout by lazy {
            device.createBindGroupLayout(
                GPUBindGroupLayoutDescriptor(
                    entries = arrayOf(
                        GPUBindGroupLayoutEntry(
                            binding = 1,
                            visibility = ShaderStage.Fragment,
                            texture = GPUTextureBindingLayout(sampleType = TextureSampleType.Float)
                        ),
                        GPUBindGroupLayoutEntry(
                            binding = 2,
                            visibility = ShaderStage.Fragment,
                            sampler = GPUSamplerBindingLayout(type = SamplerBindingType.Filtering)
                        ),
                    )
                )
            )
        }

        private val blitPipelineLayout by lazy {
            device.createPipelineLayout(
                GPUPipelineLayoutDescriptor(
                    bindGroupLayouts = arrayOf(blitBindGroupLayout),
                    immediateSize = FRAME_IMMEDIATE_BYTES,
                )
            )
        }

        // Must match [Lut3dTexture.wgsl].
        private val lutBindGroupLayout by lazy {
            device.createBindGroupLayout(
                GPUBindGroupLayoutDescriptor(
                    entries = arrayOf(
                        GPUBindGroupLayoutEntry(
                            binding = 0,
                            visibility = ShaderStage.Fragment,
                            texture = GPUTextureBindingLayout(
                                sampleType = TextureSampleType.Float,
                                viewDimension = TextureViewDimension._3D
                            )
                        ),
                        GPUBindGroupLayoutEntry(
                            binding = 1,
                            visibility = ShaderStage.Fragment,
                            sampler = GPUSamplerBindingLayout(type = SamplerBindingType.Filtering)
                        ),
                    )
                )
            )
        }

        private val blitLutPipelineLayout by lazy {
            device.createPipelineLayout(
                GPUPipelineLayoutDescriptor(
                    bindGroupLayouts = arrayOf(blitBindGroupLayout, lutBindGroupLayout),
                    immediateSize = FRAME_IMMEDIATE_BYTES,
                )
            )
        }

        private fun buildBlitPipeline(
            format: Int,
            depthStencil: GPUDepthStencilState?,
            lut: Boolean = false,
        ): GPURenderPipeline {
            val code = BLIT_SHADER + if (lut) Lut3dTexture.wgsl(1, 0) + BLIT_LUT_FS else BLIT_FS
            val shaderModule = device.createShaderModule(
                GPUShaderModuleDescriptor(shaderSourceWGSL = GPUShaderSourceWGSL(code))
            )
            return device.createRenderPipeline(
                GPURenderPipelineDescriptor(
                    vertex = GPUVertexState(
                        module = shaderModule, entryPoint = "vs_main", buffers = arrayOf(
                            GPUVertexBufferLayout(
                                arrayStride = INSTANCE_BYTES,
                                stepMode = VertexStepMode.Instance,
                                attributes = arrayOf(
                                    GPUVertexAttribute(
                                        format = VertexFormat.Float32x4,
                                        offset = 0,
                                        shaderLocation = 0
                                    )
                                )
                            )
                        )
                    ),
                    layout = if (lut) blitLutPipelineLayout else blitPipelineLayout,
                    fragment = GPUFragmentState(
                        module = shaderModule, entryPoint = "fs_main", targets = arrayOf(
                            GPUColorTargetState(
                                // Tiles hold RenderPage's output, which is premultiplied, so One
                                // rather than SrcAlpha.
                                format = format, blend = GPUBlendState(
                                    color = GPUBlendComponent(
                                        srcFactor = BlendFactor.One,
                                        dstFactor = BlendFactor.OneMinusSrcAlpha,
                                        operation = BlendOperation.Add
                                    ), alpha = GPUBlendComponent(
                                        srcFactor = BlendFactor.One,
                                        dstFactor = BlendFactor.OneMinusSrcAlpha,
                                        operation = BlendOperation.Add
                                    )
                                )
                            )
                        )
                    ),
                    primitive = GPUPrimitiveState(topology = PrimitiveTopology.TriangleList),
                    depthStencil = depthStencil,
                )
            )
        }

        /**
         * Plain blit, no stencil attachment - [blitAvailableTiles]/[renderFullyTiled]'s own pass.
         */
        private val blitPipelines = FormatKeyed { format -> buildBlitPipeline(format, null) }

        /**
         * As [blitPipeline], but always writes 1 into the stencil attachment wherever it draws -
         * for [drawCore]'s live-render callers ([draw]), which mask [RenderPage.renderFast]'s
         * per-pixel work against exactly this: a tile pixel drawn here needs no further shading
         * underneath it, so a pixel [RenderPage] would otherwise redraw stays skipped once this
         * stencil value marks it done. See [stencilViewFor].
         */
        private val blitPipelinesStencilWrite = FormatKeyed { format ->
            buildBlitPipeline(format, stencilWriteState)
        }

        private val stencilWriteState = GPUDepthStencilState(
            format = TextureFormat.Stencil8,
            depthWriteEnabled = OptionalBool.False,
            depthCompare = CompareFunction.Always,
            stencilFront = GPUStencilFaceState(
                compare = CompareFunction.Always, passOp = StencilOperation.Replace
            ),
            stencilBack = GPUStencilFaceState(
                compare = CompareFunction.Always, passOp = StencilOperation.Replace
            ),
            stencilWriteMask = 0xFF,
        )

        private val blitLutPipelines = FormatKeyed { format ->
            buildBlitPipeline(format, null, true)
        }
        private val blitLutPipelinesStencilWrite = FormatKeyed { format ->
            buildBlitPipeline(format, stencilWriteState, true)
        }
    }

    /**
     * Screens' worth of tiles to cache - the count itself follows the viewport ([budgetTiles]).
     * 1.5 is what a flat 192 came to at 1440p.
     */
    var cacheScreens = 1.5f

    // The viewport the last draw saw - what the cache is sized against.
    private var viewportWidth = 0
    private var viewportHeight = 0

    // [budgetTiles] when the atlas was built - one fixed allocation, so the budget from then on.
    private var atlasBudgetTiles = 0

    /** [cacheScreens] screens at [TILE_SIZE], ring included, within the cache byte range. */
    private fun budgetTiles(): Int {
        if (viewportWidth <= 0 || viewportHeight <= 0) return MIN_CACHE_BYTES / TILE_BYTES
        val cols = ceil(viewportWidth / TILE_SIZE.toFloat()).toInt() + 2
        val rows = ceil(viewportHeight / TILE_SIZE.toFloat()).toInt() + 2
        return (cols * rows * cacheScreens).toInt()
            .coerceIn(MIN_CACHE_BYTES / TILE_BYTES, MAX_CACHE_BYTES / TILE_BYTES)
    }

    /**
     * The size grids are cut at from now on, chosen by [reconsiderTileSize] from measured cost.
     * A grid adopts it on its next draw, through the same wipe a scale change goes through, so
     * this never disturbs one mid-gesture.
     */
    var preferredTileSize = TILE_SIZE
        set(value) {
            field = value.coerceIn(MIN_TILE_SIZE, SLAB_SIZE)
        }

    /**
     * True shows tiles as they land. False holds a grid's tiles back until every visible one is
     * ready, then shows them in one frame - for displays where each presented frame is a visible
     * refresh, like e-ink. A grid shown whole once ([PageTiles.revealed]) then updates as usual
     * until it is wiped, and keeps its tile size rather than re-cutting to a new
     * [preferredTileSize], which would hide it again. Safe from any thread.
     */
    @Volatile
    var progressive: Boolean = true
        set(value) {
            if (field == value) return
            field = value
            invalidate()
        }

    /**
     * See [TilesFilterChain]. Tiles only: the live fast path ([RenderPage.renderFast]) a pan or
     * pinch draws through stays cheap. Changing a sampler wipes every grid: cached tiles used the
     * old one.
     */
    private val tilesFilters = TilesFilterChain { replaceSampler() }

    /** See [TilesFilterChain.upscaler]. Safe from any thread. */
    var upscaler: Sampler
        get() = tilesFilters.upscaler
        set(value) {
            tilesFilters.upscaler = value
        }

    /** See [TilesFilterChain.downscaler]. Safe from any thread. */
    var downscaler: Sampler
        get() = tilesFilters.downscaler
        set(value) {
            tilesFilters.downscaler = value
        }

    /**
     * Filters over the images, whose output tiles read - see [ImageFilterChain]. A failing filter
     * wipes every grid: tiles were cut from its output.
     */
    private val imageChain = ImageFilterChain { replaceSampler() }

    /** Times every tile this generates - see [TileProfile]. */
    val profile = TileProfile()

    // Applied when tiles are blitted, not baked into them, so a change regenerates none.
    val imageLut = Lut3dTexture()

    // From [newFrame] before anything draws, when the applied table changes. Render thread.
    var onImageLutChanged: () -> Unit = {}

    // Rebuilt when [imageLut]'s view is.
    private var lutBindGroup: GPUBindGroup? = null

    /**
     * In run order - see [ImageFilterChain]. Assigning wipes every grid and cleans up dropped
     * filters. Throws for one [tileFilters] holds: a filter is bound to one chain's input and tells
     * one holder it failed. Safe from any thread; the chain takes it on the worker.
     */
    @Volatile
    var imageFilters: List<ImageFilter> = emptyList()
        set(value) {
            val previous = field
            if (previous == value) return
            require(value.none { it in tileFilters }) {
                "a filter cannot be in both imageFilters and tileFilters"
            }
            val next = value.toList()
            field = next
            replaceSampler {
                imageChain.filters = next
                previous.filter { it !in next }.forEach { it.cleanup() }
            }
        }

    /**
     * Filters run per tile, kept no longer than the tile - see [TilesFilterChain]. Assigning wipes
     * every grid and cleans up dropped filters. Throws for one [imageFilters] holds. Safe from any
     * thread; the chain takes it on the worker.
     */
    @Volatile
    var tileFilters: List<ImageFilter> = emptyList()
        set(value) {
            val previous = field
            if (previous == value) return
            require(value.none { it in imageFilters }) {
                "a filter cannot be in both imageFilters and tileFilters"
            }
            val next = value.toList()
            field = next
            replaceSampler {
                tilesFilters.filters = next
                previous.filter { it !in next }.forEach { it.cleanup() }
            }
        }

    /**
     * Apply [change] and drop every tile the outgoing sampler or filters produced. On the worker,
     * which owns both the grids and the chains' textures.
     *
     * The cost measurements go too: they timed the outgoing pipeline, and an exponential average
     * needs twenty-odd tiles per size to forget them. Kept, they mislead [reconsiderTileSize] and
     * batch pacing both ways: a cheaper pipeline reads as expensive and shrinks tiles for
     * nothing; a dearer one reads as affordable just as its first tiles land. [preferredTileSize]
     * resets too: nothing measured justifies the old size, and the grids are re-cut here anyway.
     */
    private fun replaceSampler(change: () -> Unit = {}) {
        workerScope.launch {
            // Nothing to free on a lost device, and withContext would throw.
            if (!WebGpuRenderer.isAvailable) return@launch
            // A frame may be recording with the outgoing textures.
            WebGpuRenderer.withContext {
                change()
                pages.values.forEach { releaseTiles(it) }
                // Their output is stale: the filters changed, or one failed.
                imageChain.clear()
                imageChainErrors.clear()
            }
            tileCostNs.fill(0.0)
            tileSamples.fill(0)
            probeAttempts.fill(0)
            climbedFrom = -1
            climbFailed.fill(false)
            tileOverheadNs = 0.0
            blockCostNs.fill(0.0)
            blockSamples.fill(0)
            blockProbeAttempts.fill(0)
            imageChain.blockSize = BLOCK_SIZE
            preferredTileSize = TILE_SIZE
            invalidate()
        }
    }

    /**
     * True when a filter adds passes, so a tile costs far more than [RenderPage.render] alone and
     * [probeTileSize] cannot reproduce that cost.
     */
    private val staged
        get() = imageChain.isActive || tilesFilters.isActive

    /**
     * Tile pixels per source pixel at [st]'s scale. The source is the image filters' level ([k]
     * texels per image pixel) where they apply, else the image.
     */
    private fun sourceScale(st: PageTiles): Float {
        val geometry = imageGeometry(st)
        return if (geometry != null) st.scale / geometry.k else st.scale
    }

    /** [ImageFilterChain.geometry], null where it cannot run on blocks - [imageLevels] says why. */
    private fun imageGeometry(st: PageTiles): BlockGeometry? = try {
        imageChain.geometry(st.scale)
    } catch (e: Exception) {
        null
    }

    /**
     * Whether either chain's filters run on a tile of [st]'s scale and size, so a plain one is owed
     * an upgrade once on screen. Shared by [generateTileNow] and [drawCore].
     */
    private fun filtersApply(st: PageTiles): Boolean {
        if (imageChain.runsAt(st.scale)) return true
        return tilesFilters.runsAt(st.scale) && tilesFilters.fits(
            st.tileSize,
            sourceScale(st),
            st.scale
        )
    }

    private var frame = 0L
    private var workerActive = false

    // Tiles are an optimisation over drawing pages directly: a failure in the worker is logged, never
    // left to reach the thread's uncaught exception handler and end the process.
    private val workerScope = CoroutineScope(
        WebGpuRenderer.dispatcher + SupervisorJob() +
                CoroutineExceptionHandler { _, e -> Log.e(TAG, "Tile worker failed", e) },
    )

    // Timestamp-query based GPU cost measurement for [generateTile]'s batches - null wherever the
    // adapter didn't have the feature (see WebGpuRenderer's requiredFeatures), in which case
    // batches just fall back to [TILES_PER_BATCH_FALLBACK] tiles forever.
    private val timestampQuerySet: GPUQuerySet? by lazy {
        if (!device.hasFeature(FeatureName.TimestampQuery)) return@lazy null
        device.createQuerySet(GPUQuerySetDescriptor(type = QueryType.Timestamp, count = 16))
    }

    /** [resolve] takes the query set, [result] is mapped to read it. */
    private class TimestampBuffers(val resolve: GPUBuffer, val result: GPUBuffer)

    // Recycled: two creates and a destroy per tile was a real slice of a small-tile batch.
    private val timestampPool = ArrayDeque<TimestampBuffers>()

    private fun acquireTimestampBuffers(): TimestampBuffers =
        timestampPool.removeLastOrNull() ?: TimestampBuffers(
            device.createBuffer(
                GPUBufferDescriptor(
                    size = PROFILE_QUERY_COUNT * 8L,
                    usage = BufferUsage.QueryResolve or BufferUsage.CopySrc
                )
            ),
            device.createBuffer(
                GPUBufferDescriptor(
                    size = PROFILE_QUERY_COUNT * 8L,
                    usage = BufferUsage.MapRead or BufferUsage.CopyDst
                )
            )
        )

    private fun releaseTimestampBuffers(buffers: TimestampBuffers) {
        // A batch never has more in flight than this.
        if (timestampPool.size >= MAX_TILES_PER_BATCH) {
            buffers.resolve.destroyAndRelease()
            buffers.result.destroyAndRelease()
        } else {
            timestampPool.addLast(buffers)
        }
    }

    // What recording a tile costs on the render thread - encoder, pass, submit - which the pass
    // timestamps don't span. Averaged like [tileCostNs]; 0 until the first tile lands.
    private var tileOverheadNs = 0.0

    private fun recordTileOverhead(sampleNs: Double) {
        tileOverheadNs =
            if (tileOverheadNs <= 0.0) sampleNs else tileOverheadNs * 0.8 + sampleNs * 0.2
    }

    /** A tile's whole cost to a batch: its timed pass plus [tileOverheadNs]. */
    private fun totalTileCostNs(i: Int) = tileCostNs[i] + tileOverheadNs

    // Exponential moving average of one tile's GPU render-pass duration per entry of
    // [TILE_SIZES], in nanoseconds - 0 until that size's first measurement lands.
    private val tileCostNs = DoubleArray(TILE_SIZES.size)
    private val tileSamples = IntArray(TILE_SIZES.size)

    /** Where [reconsiderTileSize]'s last step up while staged started, or -1. */
    private var climbedFrom = -1

    /** Sizes whose step up came straight back down. */
    private val climbFailed = BooleanArray(TILE_SIZES.size)

    /** Probes per size: one whose timing never lands would otherwise repeat forever. */
    private val probeAttempts = IntArray(TILE_SIZES.size)

    private fun sizeIndex(tileSize: Int) = TILE_SIZES.indexOf(tileSize)

    private fun currentTileCostNs() = tileCostNs.getOrElse(sizeIndex(preferredTileSize)) { 0.0 }

    // One block's GPU time per entry of [BLOCK_SIZES], averaged like [tileCostNs]. Apart from tile
    // costs: block size is independent of tile size, so folded in it would shrink tiles for nothing.
    private val blockCostNs = DoubleArray(BLOCK_SIZES.size)
    private val blockSamples = IntArray(BLOCK_SIZES.size)
    private val blockProbeAttempts = IntArray(BLOCK_SIZES.size)

    private fun blockIndex(blockSize: Int) = BLOCK_SIZES.indexOf(blockSize)

    private fun currentBlockCostNs() =
        blockCostNs.getOrElse(blockIndex(imageChain.blockSize)) { 0.0 }

    private fun recordBlockCost(blockSize: Int, sampleNs: Double) {
        val i = blockIndex(blockSize)
        if (i < 0 || sampleNs <= 0.0) return
        blockCostNs[i] =
            if (blockCostNs[i] <= 0.0) sampleNs else blockCostNs[i] * 0.8 + sampleNs * 0.2
        blockSamples[i]++
        reconsiderBlockSize()
    }

    /**
     * As [reconsiderTileSize], for [ImageFilterChain.blockSize]. Not frozen while filters run:
     * [probeBlockSize] runs the same chain. A change re-cuts nothing; levels are remade at the new
     * size as tiles ask.
     */
    private fun reconsiderBlockSize() {
        val current = blockIndex(imageChain.blockSize)
        if (current < 0 || blockSamples[current] < TILE_SIZE_SAMPLES) return

        if (blockCostNs[current] > BATCH_TARGET_NS && current > 0) {
            imageChain.blockSize = BLOCK_SIZES[current - 1]
            return
        }

        var best = current
        var bestCost = blockCostPerPixel(current)
        for (i in BLOCK_SIZES.indices) {
            if (i == current || blockSamples[i] < TILE_SIZE_SAMPLES) continue
            if (blockCostNs[i] > BATCH_TARGET_NS) continue
            val cost = blockCostPerPixel(i)
            if (cost * TILE_SIZE_MARGIN < bestCost) {
                best = i
                bestCost = cost
            }
        }
        imageChain.blockSize = BLOCK_SIZES[best]
    }

    /** Per output pixel: the halo, fixed per block, is what a small block wastes. */
    private fun blockCostPerPixel(i: Int) =
        blockCostNs[i] / (BLOCK_SIZES[i].toDouble() * BLOCK_SIZES[i])

    /** Blocks one step may make: a tile needing more makes them over several - see [fillBlocks]. */
    private fun blocksPerStep(): Int {
        if (timestampQuerySet == null) return MAX_BLOCKS_PER_STEP
        val cost = currentBlockCostNs()
        return if (cost <= 0.0) {
            BLOCKS_PER_STEP_FALLBACK
        } else {
            (BATCH_TARGET_NS / cost).toInt().coerceIn(1, MAX_BLOCKS_PER_STEP)
        }
    }

    /** What [step] is expected to cost a batch; unmeasured, a [TILES_PER_BATCH_FALLBACK]th of one. */
    private fun estimatedNs(step: Generated): Double {
        val unmeasured = BATCH_TARGET_NS / TILES_PER_BATCH_FALLBACK
        val blockCost = currentBlockCostNs()
        if (step.blocks > 0 && blockCost <= 0.0) return unmeasured
        val blocks = step.blocks * blockCost
        if (!step.tile) return blocks
        val tile = currentTileCostNs()
        return if (tile <= 0.0) unmeasured else tile + tileOverheadNs + blocks
    }

    /**
     * Fold one timed tile into its size's average and re-pick [preferredTileSize]. A size outside
     * [TILE_SIZES] is left alone, since nothing here can compare it.
     */
    private fun recordTileCost(tileSize: Int, sampleNs: Double) {
        val i = sizeIndex(tileSize)
        if (i < 0 || sampleNs <= 0.0) return
        tileCostNs[i] =
            if (tileCostNs[i] <= 0.0) sampleNs else tileCostNs[i] * 0.8 + sampleNs * 0.2
        tileSamples[i]++
        reconsiderTileSize()
    }

    /**
     * A size's cost per pixel - what decides, since the pixels are the work. Counts the per-tile
     * overhead, which is size-independent and so does amortise better on a big tile.
     */
    private fun costPerPixel(i: Int) =
        totalTileCostNs(i) / (TILE_SIZES[i].toDouble() * TILE_SIZES[i])

    /**
     * Pick the size whose pixels are cheapest, among those with [TILE_SIZE_SAMPLES] readings and
     * a tile inside [BATCH_TARGET_NS] - one tile is the smallest unit [schedule] can pace, so a
     * tile costing more than a batch's target is itself the hitch. A challenger needs
     * [TILE_SIZE_MARGIN] to win, since switching re-cuts every grid.
     *
     * The comparison between sizes is frozen while [staged], because the sizes are then not
     * comparable: the size in use is timed generating real tiles through the sampler, every other
     * size by [probeTileSize] without one. So the size in use reads as expensive, this switches
     * away, and the size it switches to becomes expensive in turn - and every switch re-cuts every
     * grid (see [drawCore]), which on screen is the high-quality tiles dropping out and back while
     * only the scroll moves.
     *
     * The step down is not part of that comparison and stays live while staged. It asks only
     * whether the size in use costs more than a whole batch's budget - one number, measured the
     * same way whether or not a sampler ran - and it only ever descends, so it cannot oscillate
     * the way the comparison would. Staged is where it matters most: [ImageFilterWgsl] adds a
     * compute dispatch per pass to every tile, enough for one tile to outlast its frame; frozen,
     * the renderer would have no way down from a size it cannot afford.
     */
    private fun reconsiderTileSize() {
        val current = sizeIndex(preferredTileSize)
        if (current < 0 || tileSamples[current] < TILE_SIZE_SAMPLES) return

        if (totalTileCostNs(current) > BATCH_TARGET_NS && current > 0) {
            switchTileSize(current - 1)
            return
        }

        if (staged) {
            // Frozen comparison or not, a size must be able to climb back: a slow start would
            // otherwise pin the smallest. One up holds four times the pixels; the margin covers
            // that a tile's cost isn't quite proportional to them.
            if (current < TILE_SIZES.lastIndex && !climbFailed[current] &&
                totalTileCostNs(current) * 4 * TILE_SIZE_MARGIN < BATCH_TARGET_NS
            ) {
                switchTileSize(current + 1)
                climbedFrom = current
            }
            return
        }

        var best = current
        var bestCost = costPerPixel(current)
        for (i in TILE_SIZES.indices) {
            if (i == current || tileSamples[i] < TILE_SIZE_SAMPLES) continue
            if (totalTileCostNs(i) > BATCH_TARGET_NS) continue
            val cost = costPerPixel(i)
            if (cost * TILE_SIZE_MARGIN < bestCost) {
                best = i
                bestCost = cost
            }
        }
        if (best == current) return

        // Grids re-cut on their next draw - see drawCore's invalidation.
        preferredTileSize = TILE_SIZES[best]
        invalidate()
    }

    /**
     * Moves to size [i], measured afresh: its old average would at once send it back where it came
     * from, the two sizes trading places for good.
     */
    private fun switchTileSize(i: Int) {
        // Down to where a step up started: that one didn't pay, so it isn't taken again.
        if (i == climbedFrom && i < sizeIndex(preferredTileSize)) climbFailed[i] = true
        climbedFrom = -1
        tileCostNs[i] = 0.0
        tileSamples[i] = 0
        preferredTileSize = TILE_SIZES[i]
        invalidate()
    }

    /** One cached tile: where in the [TileAtlas] it sits (packed), and when it was last drawn. */
    private class Tile(val atlasOrigin: Int) {
        var lastUsed = 0L

        /**
         * Cut off screen, so without the filters - see [generateTileNow]. [drawCore] re-cuts it once
         * visible; a tile cut on screen never is, however that went.
         */
        var plain = false
    }

    /**
     * How [generateTile] cuts a tile. [plain]: off screen, without filters, re-cut once on screen.
     * [opens]: `prepare` opens a pass to carry the opening timestamp - when it makes blocks of the
     * image filters' level, or the tiles filters run. [detailed]: the profile can split its time,
     * which it can for the tiles filters alone.
     */
    private class TileMode(val plain: Boolean, val opens: Boolean, val detailed: Boolean)

    /** An image a tile reads through its filtered level; [rect] is the part read, in image pixels. */
    private class FilteredImage(val image: Image, val level: Mipmap, val rect: ImageRect)

    /** What [imageLevels] finds for a tile. */
    private sealed class ImageLevels {
        class Ready(val k: Int, val images: List<FilteredImage>, val missing: Int) : ImageLevels()

        /** A level can't be made or held: the tile is left blank, to show it. */
        object Blank : ImageLevels()

        /** It can, but not beside this frame's: the tile is left missing, so the next frame asks again. */
        object Later : ImageLevels()
    }

    /**
     * Every tile in one texture, carved into [SLAB_SIZE] slabs. A slab holds slots of a single
     * tile size and returns to the pool once fully free, so sizes mix within one bind group.
     */
    private inner class TileAtlas(val side: Int, val format: Int) {
        val texture: GPUTexture = device.createTexture(
            GPUTextureDescriptor(
                size = GPUExtent3D(side, side),
                usage = TextureUsage.CopyDst or TextureUsage.TextureBinding,
                format = format
            )
        )

        val view: GPUTextureView = texture.createView()

        // Rendered here, then copied into the slot: compat mode has no single-slot view to
        // render into. One per size - [RenderPage] takes the target's dimensions as the tile's.
        private val scratches = HashMap<Int, GPUTexture>()
        private val scratchViews = HashMap<Int, GPUTextureView>()

        fun scratch(tileSize: Int): GPUTexture = scratches.getOrPut(tileSize) {
            device.createTexture(
                GPUTextureDescriptor(
                    size = GPUExtent3D(tileSize, tileSize),
                    usage = TextureUsage.RenderAttachment or TextureUsage.CopySrc,
                    format = format
                )
            )
        }

        fun scratchView(tileSize: Int): GPUTextureView =
            scratchViews.getOrPut(tileSize) { scratch(tileSize).createView() }

        private val slabsPerRow = side / SLAB_SIZE
        private val slabs = arrayOfNulls<Slab>(slabsPerRow * slabsPerRow)

        /** Slabs of each tile size that still have a free slot, most recently used first. */
        private val open = HashMap<Int, ArrayDeque<Slab>>()

        /** Same-sized slots; [free] holds slot indices within the slab. */
        private inner class Slab(val index: Int, val tileSize: Int) {
            val perRow = SLAB_SIZE / tileSize
            val free = IntArray(perRow * perRow) { it }
            var freeCount = free.size

            val originX = index % slabsPerRow * SLAB_SIZE
            val originY = index / slabsPerRow * SLAB_SIZE

            fun take(): Int {
                val slot = free[--freeCount]
                return pack(originX + slot % perRow * tileSize, originY + slot / perRow * tileSize)
            }

            fun give(origin: Int) {
                val slot =
                    (unpackY(origin) - originY) / tileSize * perRow + (unpackX(origin) - originX) / tileSize
                free[freeCount++] = slot
            }
        }

        /** A packed atlas position, or -1 when full - the caller drops that tile for now. */
        fun acquire(tileSize: Int): Int {
            val deque = open.getOrPut(tileSize) { ArrayDeque() }
            while (deque.isNotEmpty()) {
                val slab = deque.first()
                if (slab.freeCount > 0) {
                    val origin = slab.take()
                    if (slab.freeCount == 0) deque.removeFirst()
                    return origin
                }
                deque.removeFirst()
            }

            val index = slabs.indexOfFirst { it == null }
            if (index < 0) return -1
            val slab = Slab(index, tileSize)
            slabs[index] = slab
            val origin = slab.take()
            if (slab.freeCount > 0) deque.addFirst(slab)
            return origin
        }

        fun release(tileSize: Int, origin: Int) {
            val slab = slabs[slabIndexOf(origin)] ?: return
            slab.give(origin)
            val deque = open.getOrPut(tileSize) { ArrayDeque() }
            if (slab.freeCount == slab.free.size) {
                // Fully free: another size can claim it.
                deque.remove(slab)
                slabs[slab.index] = null
            } else if (slab.freeCount == 1) {
                deque.addFirst(slab)
            }
        }

        private fun slabIndexOf(origin: Int) =
            unpackY(origin) / SLAB_SIZE * slabsPerRow + unpackX(origin) / SLAB_SIZE

        /** Move what [scratch] of [tileSize] holds into the slot at [origin]. */
        fun copyScratchInto(encoder: GPUCommandEncoder, origin: Int, tileSize: Int) =
            encoder.copyTextureToTexture(
                GPUTexelCopyTextureInfo(texture = scratch(tileSize)),
                GPUTexelCopyTextureInfo(
                    texture = texture,
                    origin = GPUOrigin3D(x = unpackX(origin), y = unpackY(origin))
                ),
                GPUExtent3D(tileSize, tileSize)
            )

        fun destroy() {
            scratchViews.values.forEach { it.close() }
            scratchViews.clear()
            scratches.values.forEach { it.destroyAndRelease() }
            scratches.clear()
            view.close()
            texture.destroyAndRelease()
        }
    }

    // Allocated on the first tile: a viewer that never tiles pays nothing, and the viewport it
    // is sized against is known by then.
    private var atlasOrNull: TileAtlas? = null

    // Torn down in [newFrame], never here: a getter runs wherever it is first touched, which
    // can be mid-draw, holding one of the page grids the teardown destroys.
    private val atlas: TileAtlas
        get() = atlasOrNull
            ?: TileAtlas(atlasSide(), Hdr.frameFormat).also { atlasOrNull = it }

    /** Square, whole slabs, big enough for [budgetTiles] tiles of [TILE_SIZE]. */
    private fun atlasSide(): Int {
        val budget = budgetTiles()
        atlasBudgetTiles = budget
        val perSlab = (SLAB_SIZE / TILE_SIZE) * (SLAB_SIZE / TILE_SIZE)
        val slabs = (budget + perSlab - 1) / perSlab
        return ceil(sqrt(slabs.toFloat())).toInt().coerceAtLeast(1) * SLAB_SIZE
    }

    private fun newGrid(page: ImagePage.ImageSingle, pageScale: Float) =
        PageTiles(pageScale, page, preferredTileSize)

    /**
     * Release the grid longest without a draw, to get whole slabs back. Never one on screen: its
     * slots may belong to a pass still being recorded.
     */
    private fun freeColdestGrid(keep: PageTiles) {
        val victim = pages.values.firstOrNull {
            it !== keep && it.tiles.isNotEmpty() && !it.page.isOnScreen &&
                    it.tiles.values.none { t -> t.lastUsed >= frame - 1 }
        } ?: return
        releaseTiles(victim)
        victim.pending.clear()
    }

    /** Hand [st]'s tiles back to the atlas - it keeps none of its own state about them. */
    private fun releaseTiles(st: PageTiles) {
        val pool = atlasOrNull
        if (pool != null) st.tiles.values.forEach { pool.release(st.tileSize, it.atlasOrigin) }
        st.tiles.clear()
        st.instancesDirty = true
        st.sweptRange = null
        st.revealed = false
    }

    /** Packed atlas position - both halves are well inside 16 bits at any sane atlas size. */
    private fun pack(x: Int, y: Int) = (x shl 16) or y

    private fun unpackX(origin: Int) = origin ushr 16

    private fun unpackY(origin: Int) = origin and 0xFFFF

    /** One grid per whole [ImagePage.ImageSingle] - both images of a spread share it, seam baked in. */
    private class PageTiles(
        var scale: Float,
        val page: ImagePage.ImageSingle,
        /** Cut at this size until the grid is wiped, which is when it adopts a new preferred one. */
        var tileSize: Int,
    ) {
        /** [ImagePage.ImageSingle.contentVersion] the tiles were cut from. */
        var contentVersion = page.contentVersion

        val tiles = HashMap<Long, Tile>()
        val pending = HashSet<Long>()

        // Set by [placeGrid], sent by [setBlitPipeline].
        var snapX = 0f
        var snapY = 0f
        var dstW = 1f
        var dstH = 1f
        var clipL = 0f
        var clipT = 0f
        var clipR = 0f
        var clipB = 0f

        /**
         * True once a draw found every visible tile cut and none owed a re-cut; false again when
         * the grid is wiped. Until then, [progressive] false keeps the tiles off screen.
         */
        var revealed = false

        /** True once the scale has held for two consecutive frames; gates generation. */
        var stable = false

        /** Wanted range the stale sweep last ran against - see [drawCore]'s sweep. */
        var sweptRange: GridRange? = null

        // One bind group per grid, one instance per tile. Instances change only when a tile is
        // generated or dropped, never as the grid moves - the immediates' snap does that.
        var bindGroup: GPUBindGroup? = null
        var instances: GPUBuffer? = null
        var instanceCapacity = 0
        var instanceCount = 0
        var instancesDirty = true

        /**
         * This page's exact, unrounded vertical offset from the grid's shared anchor - see
         * [draw]'s continuous overload. Always 0 for the paged overload.
         *
         * Also doubles as a staleness key, compared each call like [scale]: a changed offset at
         * fixed scale means the page's document position shifted, so existing tiles no longer
         * agree with where it sits.
         */
        var centerYOffset = 0f

        // The strictly visible tile range as of the last draw, in tile coordinates. The worker
        // prioritises against it at pull time, so a pan mid-fill redirects generation without
        // touching the queue.
        var txMin = 0
        var txMax = -1
        var tyMin = 0
        var tyMax = -1

        val destroyed get() = page.destroyed

        fun destroyAll(atlas: TileAtlas?) {
            tiles.values.forEach { atlas?.release(tileSize, it.atlasOrigin) }
            tiles.clear()
            pending.clear()
            instances?.destroyAndRelease()
            instances = null
            instanceCapacity = 0
            instanceCount = 0
            bindGroup?.close()
            bindGroup = null
            revealed = false
        }
    }

    /** One tile of work for the shared worker - see [schedule]/[nextRequest]. */
    private class Request(
        val state: PageTiles, val tx: Int, val ty: Int, val onScreen: Boolean
    )

    // Access-ordered so getOrPut's read-then-maybe-write always moves the touched page to the
    // end (most recently drawn), whether or not it was already present - see RETAIN_MARGIN.
    private val pages = LinkedHashMap<ImagePage.ImageSingle, PageTiles>(16, 0.75f, true)

    private fun key(tx: Int, ty: Int) = (tx.toLong() shl 32) or (ty.toLong() and 0xFFFFFFFFL)

    // Render thread only.
    private val frameBytes =
        ByteBuffer.allocateDirect(FRAME_IMMEDIATE_BYTES).order(ByteOrder.nativeOrder())

    private fun setBlitPipeline(
        pass: GPURenderPassEncoder, format: Int, st: PageTiles, stencilWrite: Boolean
    ) {
        var useLut = true
        st.page.forEachImage { image, _, _ -> if (!image.useImageLut) useLut = false }
        val lut = if (useLut) RenderPage.lut else null
        val pipelines = when {
            lut == null && stencilWrite -> blitPipelinesStencilWrite
            lut == null -> blitPipelines
            stencilWrite -> blitLutPipelinesStencilWrite
            else -> blitLutPipelines
        }
        pass.setPipeline(pipelines[format])
        val bytes = frameBytes
        bytes.clear()
        bytes.putFloat(st.snapX).putFloat(st.snapY)
        bytes.putFloat(st.dstW).putFloat(st.dstH)
        bytes.putFloat(st.clipL).putFloat(st.clipT).putFloat(st.clipR).putFloat(st.clipB)
        bytes.putFloat(st.tileSize.toFloat()).putFloat(atlas.side.toFloat())
        // Struct padding.
        bytes.putFloat(0f).putFloat(0f)
        bytes.flip()
        pass.setImmediates(0, bytes)
        if (lut == null) return
        val group = lutBindGroup ?: device.createBindGroup(
            GPUBindGroupDescriptor(
                layout = lutBindGroupLayout, entries = arrayOf(
                    GPUBindGroupEntry(0, textureView = lut.view!!),
                    GPUBindGroupEntry(1, sampler = lut.sampler),
                )
            )
        ).also { lutBindGroup = it }
        pass.setBindGroup(1, group)
    }

    // Ring-buffered, not one shared texture: reusing a single stencil texture every frame would
    // force the GPU to serialize each frame's clear/write against the previous frame's, still
    // in flight since render() only serializes encoding - silently undoing the swapchain's own
    // buffering and showing up as tiles trailing the current pan/scroll position.
    private val stencilTextures = arrayOfNulls<GPUTexture>(STENCIL_BUFFER_COUNT)
    private val stencilViews = arrayOfNulls<GPUTextureView>(STENCIL_BUFFER_COUNT)
    private var stencilWidth = 0
    private var stencilHeight = 0

    /**
     * Stencil-only attachment matching [dst]'s size, shared by every live-render pass this frame
     * so [blitPipelineStencilWrite] can mark tile-covered pixels and [RenderPage]'s masked
     * variants can skip re-shading them. Rotates across [STENCIL_BUFFER_COUNT] textures by
     * [frame] so a still-in-flight previous frame never shares one with this one.
     */
    fun stencilViewFor(dst: GPUTexture): GPUTextureView {
        if (stencilWidth != dst.width || stencilHeight != dst.height) {
            stencilWidth = dst.width
            stencilHeight = dst.height
            for (i in 0 until STENCIL_BUFFER_COUNT) {
                stencilViews[i]?.close()
                stencilTextures[i]?.destroyAndRelease()
                val texture = device.createTexture(
                    GPUTextureDescriptor(
                        usage = TextureUsage.RenderAttachment or TextureUsage.TransientAttachment,
                        size = GPUExtent3D(dst.width, dst.height),
                        format = TextureFormat.Stencil8,
                    )
                )
                stencilTextures[i] = texture
                stencilViews[i] = texture.createView()
            }
        }
        return stencilViews[(frame % STENCIL_BUFFER_COUNT).toInt()]!!
    }

    /**
     * Advance the frame counter and drop tiles for any page the app has since evicted. Called
     * once at the top of every rendered frame, including ones that don't draw tiles at all (e.g.
     * a page transition), so a destroyed page's textures are freed right away.
     */
    fun newFrame() {
        frame++
        atlasOrNull?.let { pool ->
            retired.forEach { (tileSize, origin) ->
                pool.release(
                    tileSize,
                    origin
                )
            }
        }
        retired.clear()
        ErrorScopes.pump()
        imageChain.prune()

        val drawnLut = imageLut.applied
        if (imageLut.prepare()) {
            lutBindGroup?.close()
            lutBindGroup = null
        }
        RenderPage.lut = if (imageLut.applied != null) imageLut else null
        if (imageLut.applied !== drawnLut) onImageLutChanged()

        // Every cached tile is at the old format, so an HDR change takes the whole atlas. Here
        // only: this is the one point in a frame where no caller holds a grid, and tearing one
        // down mid-draw leaves the caller drawing from destroyed buffers - a segfault.
        atlasOrNull?.let { existing ->
            if (existing.format != Hdr.frameFormat) {
                pages.values.forEach { it.destroyAll(existing) }
                pages.clear()
                existing.destroy()
                atlasOrNull = null
            }
        }

        if (pages.isEmpty()) return
        val it = pages.iterator()
        while (it.hasNext()) {
            val st = it.next().value
            if (st.destroyed) {
                dropPage(st)
                it.remove()
            }
        }
    }

    /**
     * Slots of plain tiles their re-cut replaced, as (tile size, origin), freed at the next [newFrame]:
     * a pass recorded but not yet submitted - [renderFullyTiled]'s - may still read them.
     */
    private val retired = ArrayList<Pair<Int, Int>>()

    /** Free [st]'s tiles, and its images' filtered levels; the caller removes it from [pages]. */
    private fun dropPage(st: PageTiles) {
        st.destroyAll(atlasOrNull)
        // Only images no other grid draws: a spread shares its pages' images.
        val kept = HashSet<Image>()
        for (other in pages.values) {
            if (other !== st) other.page.forEachImage { image, _, _ -> kept.add(image) }
        }
        st.page.forEachImage { image, _, _ -> if (image !in kept) imageChain.forget(image) }
    }

    /**
     * Left/right screen-pixel extent of [page] from its own anchor (x=0), in [pageScale] units.
     *
     * Not symmetric halves of [ImagePage.width] for an [ImagePage.ImageSpread]: each side extends
     * outward by its *own* width, so the anchor sits at the seam rather than the centre of the
     * combined footprint - they differ whenever the sides do, e.g. a cover with no partner. A
     * plain [ImagePage.ImageSingle] has no seam, so its extent stays symmetric.
     */
    private fun pageHorizontalExtent(
        page: ImagePage.ImageSingle, pageScale: Float
    ): Pair<Float, Float> {
        val (leftWidth, rightWidth) = page.horizontalExtent()
        return pageScale * leftWidth to pageScale * rightWidth
    }

    /**
     * [pageScale]/[anchorX]/[anchorY] a paged overload resolves its (x, y, scale) placement to.
     * [pinned] when that placement is the home animation's target rather than where the page
     * actually is this frame - tiles then can't stand in for a live draw of it.
     */
    private class PagedAnchor(
        val pageScale: Float, val anchorX: Float, val anchorY: Float, val pinned: Boolean
    )

    private fun pagedAnchor(
        page: ImagePage.ImageSingle, dst: GPUTexture, x: Float, y: Float, scale: Float
    ): PagedAnchor {
        // Pin the grid to the animation's target while actually scale-animating home, so
        // drawCore wipes it once instead of every interpolated frame. Gated on isScaleAnimating,
        // not just the target being homeScale, since a pure-position animateTo at a constant
        // homeScale never sets that flag - pinning then would just freeze the grid for nothing.
        val goingHome = page.isScaleAnimating && page.animationTargetScale == page.homeScale
        val effectivePageX = if (goingHome) page.animationTargetX ?: page.x else page.x
        val effectivePageY = if (goingHome) page.animationTargetY ?: page.y else page.y
        val effectivePageScale = if (goingHome) page.homeScale else page.scale
        val pageScale = effectivePageScale * scale
        val anchorX =
            dst.width / 2f + pageScale * ((effectivePageX + x + WebGpuRenderer.offsetX) * dst.width)
        val anchorY =
            dst.height / 2f + pageScale * ((effectivePageY + y + WebGpuRenderer.offsetY) * dst.height)
        return PagedAnchor(pageScale, anchorX, anchorY, goingHome)
    }

    /**
     * Shared placement math for a page's tile grid at [anchorX]/[anchorY], scaled by [pageScale]:
     * the tile region the viewport (plus a one-tile margin) wants, clipped to the page's own
     * extent, and the snapped clip rect the shader clamps blits to. A [crop], in the page's
     * pixels, narrows both to it: nothing outside is generated or drawn.
     *
     * One definition shared by [drawCore], [availableTileKeys] and [prewarm] - each used to carry
     * its own copy, which is how [prewarm] ended up silently missing a step the others had.
     */
    private class GridPlacement(
        val ts: Float,
        val snapX: Float,
        val snapY: Float,
        val clipL: Float,
        val clipT: Float,
        val clipR: Float,
        val clipB: Float,
        val wantL: Float,
        val wantR: Float,
        val wantT: Float,
        val wantB: Float
    )

    private fun gridPlacement(
        page: ImagePage.ImageSingle,
        dst: GPUTexture,
        anchorX: Float,
        anchorY: Float,
        centerYOffset: Float,
        pageScale: Float,
        tileSize: Int,
        crop: Rect? = null,
    ): GridPlacement? {
        val ts = tileSize.toFloat()
        // Drawn extent from the page's centre, narrowed to [crop].
        var (left, right) = pageHorizontalExtent(page, pageScale)
        var top = pageScale * page.height / 2f
        var bottom = top
        if (crop != null) {
            left = min(left, pageScale * (page.width / 2f - crop.left))
            right = min(right, pageScale * (crop.right - page.width / 2f))
            top = min(top, pageScale * (page.height / 2f - crop.top))
            bottom = min(bottom, pageScale * (crop.bottom - page.height / 2f))
        }
        if (left + right <= 0f || top + bottom <= 0f) return null

        val wantL = max(-anchorX - ts, -left)
        val wantR = min(dst.width - anchorX + ts, right)
        val wantT = max(-anchorY - ts, centerYOffset - top)
        val wantB = min(dst.height - anchorY + ts, centerYOffset + bottom)

        val snapX = round(anchorX)
        val snapY = round(anchorY)
        val clipL = snapX - left
        val clipT = snapY + centerYOffset - top
        val clipR = snapX + right
        val clipB = snapY + centerYOffset + bottom

        return GridPlacement(
            ts, snapX, snapY, clipL, clipT, clipR, clipB, wantL, wantR, wantT, wantB
        )
    }

    /** The (tx, ty) index bounds of what [gp] wants, and what [drawCore] tests keys against. */
    private class GridRange(val tx0: Int, val tx1: Int, val ty0: Int, val ty1: Int) {
        fun holds(tileKey: Long): Boolean {
            val tx = (tileKey shr 32).toInt()
            val ty = tileKey.toInt()
            return tx in tx0..tx1 && ty in ty0..ty1
        }

        fun same(other: GridRange) =
            tx0 == other.tx0 && tx1 == other.tx1 && ty0 == other.ty0 && ty1 == other.ty1
    }

    private fun wantedTileRange(gp: GridPlacement): GridRange {
        val ts = gp.ts
        return GridRange(
            floor(gp.wantL / ts).toInt(),
            ceil(gp.wantR / ts).toInt() - 1,
            floor(gp.wantT / ts).toInt(),
            ceil(gp.wantB / ts).toInt() - 1
        )
    }

    /**
     * Every (tx, ty) in [r], visible or not - the one definition [drawCore] and
     * [availableTileKeys] share, so they can't drift apart the way [gridPlacement]'s own doc
     * describes [prewarm] once doing. Visibility is a separate per-tile question - see
     * [tileVisible].
     */
    private inline fun forEachTile(r: GridRange, action: (txi: Int, tyi: Int) -> Unit) {
        for (tyi in r.ty0..r.ty1) {
            for (txi in r.tx0..r.tx1) {
                action(txi, tyi)
            }
        }
    }

    /** True if tile ([txi], [tyi]) of [gp]'s grid actually overlaps [dst]'s visible bounds. */
    private fun tileVisible(gp: GridPlacement, dst: GPUTexture, txi: Int, tyi: Int): Boolean {
        val ts = gp.ts
        val px = gp.snapX + txi * ts
        val py = gp.snapY + tyi * ts
        val visL = max(gp.clipL, 0f)
        val visT = max(gp.clipT, 0f)
        val visR = min(gp.clipR, dst.width.toFloat())
        val visB = min(gp.clipB, dst.height.toFloat())
        return px < visR && px + ts > visL && py < visB && py + ts > visT
    }

    /** As [PagedAnchor], for the continuous overloads; also carries [centerYOffset]. */
    private class ContinuousAnchor(
        val pageScale: Float, val anchorX: Float, val anchorY: Float, val centerYOffset: Float
    )

    private fun continuousAnchor(
        page: ImagePage.ImageSingle,
        dst: GPUTexture,
        cameraDocY: Float,
        docTop: Float,
        pageHeight: Float,
        viewerOffsetX: Float,
        scale: Float,
        crop: Rect? = null,
    ): ContinuousAnchor? {
        if (page.width <= 0) return null
        // [crop], when given, is what fills the viewer's width; the page centres off it.
        val cropWidth = crop?.width() ?: page.width
        if (cropWidth <= 0) return null
        val pageScaleAtZoom1 = dst.width / cropWidth.toFloat()
        val pageScale = pageScaleAtZoom1 * scale
        val shiftX = crop?.let { pageScaleAtZoom1 * (page.width / 2f - it.exactCenterX()) }
            ?: 0f
        val shiftY = crop?.let { pageScaleAtZoom1 * (page.height / 2f - it.exactCenterY()) }
            ?: 0f
        val anchorX = dst.width / 2f +
                scale * (viewerOffsetX * dst.width + WebGpuRenderer.offsetX * dst.width + shiftX)
        val anchorY =
            dst.height / 2f - scale * cameraDocY + scale * WebGpuRenderer.offsetY * dst.height
        // The caller's measured height, never one derived here - the two must agree exactly.
        val centerYOffset = scale * (docTop + pageHeight / 2f + shiftY)
        return ContinuousAnchor(pageScale, anchorX, anchorY, centerYOffset)
    }

    /**
     * The set of (tx, ty) grid keys [page]'s tile cache currently has cached and visible within
     * [dst] - lets a caller like [Transition] track exactly what it's already blitted and detect
     * when something new lands, instead of re-deriving a "done yet" boolean that has to agree
     * with [drawCore] (see [forEachTile]'s doc). Null if the page isn't drawable.
     */
    fun availableTileKeys(page: ImagePage.ImageSingle, dst: GPUTexture): Set<Long>? {
        if (page.destroyed || !page.highQuality) return null
        if (!page.hasUploadedImage) return null

        val st = pages[page] ?: return null
        val a = pagedAnchor(page, dst, 0f, 0f, 1f)
        if (st.contentVersion != page.contentVersion || st.scale != a.pageScale ||
            st.tileSize != preferredTileSize
        ) return emptySet()
        val gp =
            gridPlacement(page, dst, a.anchorX, a.anchorY, 0f, a.pageScale, st.tileSize)
                ?: return null
        if (gp.wantL >= gp.wantR || gp.wantT >= gp.wantB) return emptySet()

        val keys = HashSet<Long>()
        forEachTile(wantedTileRange(gp)) { txi, tyi ->
            if (tileVisible(gp, dst, txi, tyi)) {
                val tkey = key(txi, tyi)
                if (st.tiles.containsKey(tkey)) keys.add(tkey)
            }
        }
        return keys
    }

    /**
     * Ensure [page]'s tile grid exists and enqueue its missing tiles, without blitting anything -
     * for getting a page sharp before it's on screen (the paged viewer's next page). Always at
     * the page's own home position, since it has no live pan/zoom yet.
     *
     * Deliberately leaves [PageTiles.txMin]/etc at their empty default, so [nextRequest] always
     * ranks this grid's tiles as "off-screen", behind whichever page is genuinely being drawn.
     * Doesn't run [draw]'s retain-window trim either; that catches this grid once the page it
     * displaced becomes current and calls [draw] again.
     */
    fun prewarm(page: ImagePage.ImageSingle, dst: GPUTexture) {
        if (page.destroyed || !page.highQuality) return
        if (!page.hasUploadedImage) return

        viewportWidth = dst.width
        viewportHeight = dst.height

        val a = pagedAnchor(page, dst, 0f, 0f, 1f)
        val st = pages.getOrPut(page) { newGrid(page, a.pageScale) }

        val version = page.contentVersion
        if (st.scale != a.pageScale || st.tileSize != preferredTileSize ||
            st.contentVersion != version
        ) {
            releaseTiles(st)
            st.pending.clear()
            st.scale = a.pageScale
            st.tileSize = preferredTileSize
            st.contentVersion = version
        }
        st.stable = true

        val gp = gridPlacement(page, dst, a.anchorX, a.anchorY, 0f, a.pageScale, st.tileSize)
            ?: return
        if (gp.wantL >= gp.wantR || gp.wantT >= gp.wantB) return

        val wanted = wantedTileRange(gp)
        val tx0 = wanted.tx0
        val tx1 = wanted.tx1
        val ty0 = wanted.ty0
        val ty1 = wanted.ty1

        // Captured before the loop below can add to it, so this only fires the one time pending
        // work actually starts for this grid - every later call while it's still filling in finds
        // pending already non-empty and stays quiet.
        val alreadyPrewarming = st.pending.isNotEmpty()

        var missing = 0
        for (tyi in ty0..ty1) {
            for (txi in tx0..tx1) {
                val tile = st.tiles[key(txi, tyi)]
                if (tile != null) tile.lastUsed = frame else missing++
            }
        }
        if (missing == 0) return

        var total = 0L
        for (other in pages.values) total += other.tiles.size * tileBytes(other)
        if (total + missing * tileBytes(st) > maxTileBytes) return

        var added = false
        for (tyi in ty0..ty1) {
            for (txi in tx0..tx1) {
                val tkey = key(txi, tyi)
                if (!st.tiles.containsKey(tkey)) {
                    st.pending.add(tkey)
                    added = true
                }
            }
        }
        if (added) {
            if (!alreadyPrewarming) Log.d(TAG, "Pre-warming next page tiles ${pageId(page)}")
            schedule()
        }
    }

    /**
     * Blit [page]'s cached tiles and enqueue the missing ones - the paged viewer's placement, via
     * its own [page]-relative (x, y).
     *
     * [ImagePage.Render] pages are drawn by their own override and animated pages swap images per frame -
     * neither worth the cache's sharpness, so both stay on the fast/plain path instead via
     * [ImagePage.ImageSingle.highQuality] being false.
     */
    fun draw(
        pass: GPURenderPassEncoder,
        page: ImagePage.ImageSingle,
        dst: GPUTexture,
        x: Float,
        y: Float,
        scale: Float
    ): Boolean {
        val a = pagedAnchor(page, dst, x, y, scale)
        val covered = drawCore(
            pass,
            page,
            dst,
            a.anchorX,
            a.anchorY,
            0f,
            a.pageScale,
            page.isScaleAnimating,
            applyRetainWindow = true,
            useStencilMask = true
        )
        // Never "covered" off a pinned grid: it sits at the animation's target, so the live draw
        // is the only thing showing the page where it is mid-animation.
        return covered && !a.pinned
    }

    /**
     * Blit [page]'s cached tiles and enqueue the missing ones - the continuous viewer's
     * placement, via [cameraDocY] (the camera's document position), [docTop] (this page's own
     * top) and [pageHeight] (its content height - all in screen pixels at zoom 1).
     *
     * Rounds only the shared *camera* anchor, leaving each page's own offset from it exact -
     * unlike the paged overload, several pages can draw through here in the same frame, and
     * independently rounding each one's own anchor could leave adjacent grids disagreeing by a
     * pixel at their shared seam (`round(a) + b` isn't generally `round(a + b)`). Recomputing the
     * same camera anchor from the same inputs every caller passes keeps it identical regardless
     * of which page is drawing, so two adjacent pages' clip boundaries always agree exactly.
     */
    fun draw(
        pass: GPURenderPassEncoder,
        page: ImagePage.ImageSingle,
        dst: GPUTexture,
        cameraDocY: Float,
        docTop: Float,
        pageHeight: Float,
        viewerOffsetX: Float,
        scale: Float,
        suppressGeneration: Boolean,
        crop: Rect? = null,
    ): Boolean {
        val a = continuousAnchor(
            page, dst, cameraDocY, docTop, pageHeight, viewerOffsetX, scale, crop
        ) ?: return false
        return drawCore(
            pass,
            page,
            dst,
            a.anchorX,
            a.anchorY,
            a.centerYOffset,
            a.pageScale,
            suppressGeneration,
            applyRetainWindow = false,
            useStencilMask = true,
            crop = crop,
        )
    }

    /**
     * Shared blit-and-enqueue core for both [draw] overloads. [anchorX]/[anchorY] are the
     * (unrounded) anchor either overload computed; [centerYOffset] is this page's own exact,
     * unrounded vertical offset from that anchor - zero for the paged overload. See
     * [PageTiles.centerYOffset] for why comparing it against the stored value also catches a
     * changed document position.
     */
    private fun drawCore(
        pass: GPURenderPassEncoder,
        page: ImagePage.ImageSingle,
        dst: GPUTexture,
        anchorX: Float,
        anchorY: Float,
        centerYOffset: Float,
        pageScale: Float,
        suppressGeneration: Boolean,
        applyRetainWindow: Boolean,
        useStencilMask: Boolean = false,
        crop: Rect? = null,
    ): Boolean {
        if (page.destroyed || !page.highQuality) return false
        if (!page.hasUploadedImage) return false

        // Unlike drawGridForFullPage, nothing else on this path reports the draw to Hdr.
        page.currentImage?.let { if (it.isHdr) Hdr.noteHdrDrawn(it, it.hdrHeadroom) }

        viewportWidth = dst.width
        viewportHeight = dst.height

        val st = pages.getOrPut(page) { newGrid(page, pageScale) }

        if (applyRetainWindow) {
            // getOrPut just moved page to the end of this access-ordered map - trim the front
            // (least recently drawn) down to the grace window. A page turn animates via
            // Transition's own cache, never this one, so anything evicted here isn't on screen.
            while (pages.size > RETAIN_MARGIN) {
                val eldest = pages.entries.iterator()
                val entry = eldest.next()
                dropPage(entry.value)
                eldest.remove()
            }
        }

        val version = page.contentVersion
        // Held back, a re-cut shown grid would drop to the fast path until refilled: two more
        // refreshes for nothing the eye asked for. It keeps its size; the next grid takes the new one.
        val recut = st.tileSize != preferredTileSize && (progressive || !st.revealed)
        if (st.scale != pageScale || st.centerYOffset != centerYOffset ||
            recut || st.contentVersion != version
        ) {
            // A changed centerYOffset at fixed scale means a placeholder corrected its guessed
            // height - invalidate the same way a scale change does. A changed preferred size
            // rides the same path: this is the one moment a grid can be re-cut for free.
            releaseTiles(st)
            st.pending.clear()
            st.scale = pageScale
            st.tileSize = preferredTileSize
            st.contentVersion = version
            st.stable = false
            invalidate()
        } else {
            // Two frames landing on the same scale isn't enough proof of settling while a
            // gesture/animation is still actively driving it.
            st.stable = !suppressGeneration
        }
        // Recomputed every call regardless of whether it actually changed - see the field's own
        // doc for why that's safe. generate() has no other way to reach this value.
        st.centerYOffset = centerYOffset

        val gp =
            gridPlacement(page, dst, anchorX, anchorY, centerYOffset, pageScale, st.tileSize, crop)
        if (gp == null) {
            st.pending.clear()
            return false
        }
        // Off screen: nothing to draw, so nothing for tiles to be missing either.
        if (gp.wantL >= gp.wantR || gp.wantT >= gp.wantB) {
            st.pending.clear()
            return true
        }

        val ts = gp.ts

        // In tile coordinates, unlike wantT/wantB - not offset by centerYOffset, since a tile's
        // blit position is snapY + ty*ts regardless of which page it belongs to.
        st.txMin = floor(-anchorX / ts).toInt()
        st.txMax = ceil((dst.width - anchorX) / ts).toInt() - 1
        st.tyMin = floor(-anchorY / ts).toInt()
        st.tyMax = ceil((dst.height - anchorY) / ts).toInt() - 1

        placeGrid(st, dst, gp.snapX, gp.snapY, gp.clipL, gp.clipT, gp.clipR, gp.clipB)

        val wanted = wantedTileRange(gp)

        // Bookkeeping only: the blit is one instanced draw below, and the shader clamps each tile
        // to the clip rect. Coverage still asks tileVisible - a missing tile nobody sees is fine.
        var covered = true
        // A visible plain tile still owed its filtered re-cut - see [progressive].
        var owed = false
        forEachTile(wanted) { txi, tyi ->
            val tkey = key(txi, tyi)
            val tile = st.tiles[tkey]
            if (tile != null) {
                tile.lastUsed = frame
                // Cut plain off screen; now visible, so queued for the worker to re-cut - see
                // [Tile.plain]. Drawn until the replacement lands: dropping it now would leave a hole
                // this draw reports as covered.
                if (tile.plain && st.stable && tileVisible(gp, dst, txi, tyi) && filtersApply(st)) {
                    st.pending.add(tkey)
                    owed = true
                }
            } else {
                if (st.stable) st.pending.add(tkey)
                if (covered && tileVisible(gp, dst, txi, tyi)) covered = false
            }
        }

        if (covered && !owed) st.revealed = true
        // Held back: not covered, so the caller draws the whole page the fast way instead.
        val shown = progressive || st.revealed
        if (shown) drawInstanced(pass, dst.format, st, useStencilMask)

        // Drop what fell outside the wanted range, else a page scrolling past keeps accumulating
        // tiles. Only when the range moved - a scroll crosses a tile boundary every tile-size
        // pixels, so most frames skip both walks. evict()'s staleness guard, so a tile the last
        // frame or two drew waits for the next move.
        if (st.sweptRange?.same(wanted) != true) {
            st.sweptRange = wanted
            st.pending.retainAll { wanted.holds(it) }
            val staleIt = st.tiles.entries.iterator()
            while (staleIt.hasNext()) {
                val (k, t) = staleIt.next()
                if (!wanted.holds(k) && t.lastUsed < frame - 1) {
                    atlasOrNull?.release(st.tileSize, t.atlasOrigin)
                    staleIt.remove()
                    st.instancesDirty = true
                }
            }
        }

        if (st.pending.isNotEmpty()) schedule()
        return covered && shown
    }

    /**
     * Blit [st]'s cached tiles as one instanced draw. A frame that only moved the grid uploads
     * nothing - that goes in the immediates.
     */
    private fun drawInstanced(
        pass: GPURenderPassEncoder, format: Int, st: PageTiles, useStencilMask: Boolean
    ) {
        if (st.instancesDirty) uploadInstances(st)
        val instances = st.instances ?: return
        if (st.instanceCount == 0) return

        setBlitPipeline(pass, format, st, useStencilMask)
        if (useStencilMask) pass.setStencilReference(1)
        pass.setBindGroup(0, st.bindGroup ?: gridBindGroup(st).also { st.bindGroup = it })
        pass.setVertexBuffer(0, instances)
        pass.draw(6, st.instanceCount)
    }

    /**
     * Blit exactly [keys] - for a caller that just generated tiles into a pass that already drew
     * the rest. Its own buffer: Dawn keeps a destroyed one alive until its commands retire.
     */
    private fun drawTiles(
        pass: GPURenderPassEncoder,
        format: Int,
        st: PageTiles,
        keys: List<Long>
    ) {
        var count = 0
        for (tkey in keys) if (st.tiles.containsKey(tkey)) count++
        if (count == 0) return

        val bytes = scratch(count * INSTANCE_BYTES.toInt())
        for (tkey in keys) {
            val tile = st.tiles[tkey] ?: continue
            tile.lastUsed = frame
            bytes.putFloat((tkey shr 32).toInt().toFloat())
            bytes.putFloat(tkey.toInt().toFloat())
            bytes.putFloat(unpackX(tile.atlasOrigin).toFloat())
            bytes.putFloat(unpackY(tile.atlasOrigin).toFloat())
        }
        bytes.flip()

        val instances = device.createBuffer(
            GPUBufferDescriptor(
                size = count * INSTANCE_BYTES,
                usage = BufferUsage.Vertex or BufferUsage.CopyDst
            )
        )
        device.queue.writeBuffer(instances, 0, bytes.slice())

        setBlitPipeline(pass, format, st, false)
        pass.setBindGroup(0, st.bindGroup ?: gridBindGroup(st).also { st.bindGroup = it })
        pass.setVertexBuffer(0, instances)
        pass.draw(6, count)
        instances.destroyAndRelease()
    }

    // Reused staging, render thread only. writeBuffer sends the whole capacity: pass slice().
    private var scratchBytes: ByteBuffer =
        ByteBuffer.allocateDirect(4096).order(ByteOrder.nativeOrder())

    private fun scratch(size: Int): ByteBuffer {
        if (scratchBytes.capacity() < size) {
            scratchBytes = ByteBuffer.allocateDirect(Integer.highestOneBit(size - 1) shl 1)
                .order(ByteOrder.nativeOrder())
        }
        scratchBytes.clear()
        return scratchBytes
    }

    private fun gridBindGroup(st: PageTiles): GPUBindGroup = device.createBindGroup(
        GPUBindGroupDescriptor(
            layout = blitBindGroupLayout, entries = arrayOf(
                GPUBindGroupEntry(1, textureView = atlas.view),
                GPUBindGroupEntry(2, sampler = blitSampler),
            )
        )
    )

    /**
     * Rewrite [st]'s instance data - grid coordinate and atlas position per tile. Called from the
     * draw, so a batch landing several tiles uploads once.
     */
    private fun uploadInstances(st: PageTiles) {
        st.instancesDirty = false
        st.instanceCount = st.tiles.size
        if (st.instanceCount == 0) return

        if (st.instanceCapacity < st.instanceCount) {
            // Rounded up so filling in tile by tile doesn't reallocate on every one.
            val capacity = (st.instanceCount + 63) / 64 * 64
            st.instances?.destroyAndRelease()
            st.instances = device.createBuffer(
                GPUBufferDescriptor(
                    size = capacity * INSTANCE_BYTES,
                    usage = BufferUsage.Vertex or BufferUsage.CopyDst
                )
            )
            st.instanceCapacity = capacity
        }

        val bytes = scratch(st.instanceCount * INSTANCE_BYTES.toInt())
        for ((tkey, tile) in st.tiles) {
            bytes.putFloat((tkey shr 32).toInt().toFloat())
            bytes.putFloat(tkey.toInt().toFloat())
            bytes.putFloat(unpackX(tile.atlasOrigin).toFloat())
            bytes.putFloat(unpackY(tile.atlasOrigin).toFloat())
        }
        bytes.flip()
        device.queue.writeBuffer(st.instances!!, 0, bytes.slice())
    }

    /**
     * Time one throwaway tile at an unmeasured size, so [reconsiderTileSize] has something to
     * compare. Null when there is nothing to probe - including without timestamp queries, where
     * no size can be measured and [preferredTileSize] stays where it started.
     */
    private fun probeTileSize(measurementScope: CoroutineScope): Job? {
        // Nothing to choose between while [reconsiderTileSize] is frozen, and this would measure
        // the wrong thing anyway: it renders the plain way, with no sampler in the middle.
        if (staged) return null
        val queries = timestampQuerySet ?: return null
        val index = TILE_SIZES.indices.firstOrNull {
            TILE_SIZES[it] != preferredTileSize && tileSamples[it] < TILE_SIZE_SAMPLES &&
                    probeAttempts[it] < 2 * TILE_SIZE_SAMPLES
        } ?: return null
        val tileSize = TILE_SIZES[index]

        // The most recently drawn grid: the page on screen is the cost that matters.
        val st = pages.values.lastOrNull {
            it.stable && !it.destroyed && it.page.hasUploadedImage
        } ?: return null

        probeAttempts[index]++
        val pool = atlas
        val timing = acquireTimestampBuffers()
        val encoder = device.createCommandEncoder()
        val pass = encoder.beginRenderPass(
            clearedColorPass(
                pool.scratchView(tileSize), timestampWrites = GPUPassTimestampWrites(
                    queries, beginningOfPassWriteIndex = 0, endOfPassWriteIndex = 1
                )
            )
        )
        try {
            // The grid's anchor tile - the middle of the page, as representative as this gets.
            renderTileContent(st, 0, 0, tileSize, pass, pool.scratch(tileSize))
        } finally {
            pass.endAndRelease()
        }

        encoder.resolveQuerySet(queries, 0, 2, timing.resolve, 0)
        encoder.copyBufferToBuffer(timing.resolve, 0, timing.result, 0, 16)
        device.queue.submitAndRelease(encoder)

        // Not a run: a probe times an unused size, not a held tile.
        return measurementScope.launch { measureTileGpuTime(timing, tileSize, null, 16, 0, 0) }
    }

    /**
     * Times a throwaway block at an unmeasured size for [reconsiderBlockSize]. Null when nothing is
     * left to probe or there are no timestamp queries.
     */
    private fun probeBlockSize(measurementScope: CoroutineScope): Job? {
        if (!imageChain.isActive) return null
        val queries = timestampQuerySet ?: return null
        val st = pages.values.lastOrNull {
            it.stable && !it.destroyed && it.page.hasUploadedImage && imageChain.runsAt(it.scale)
        } ?: return null
        var image: Image? = null
        st.page.forEachImage { candidate, _, _ ->
            if (image == null && candidate.mipmaps.isNotEmpty()) image = candidate
        }
        val probed = image ?: return null

        while (true) {
            val index = BLOCK_SIZES.indices.firstOrNull {
                BLOCK_SIZES[it] != imageChain.blockSize && blockSamples[it] < TILE_SIZE_SAMPLES &&
                        blockProbeAttempts[it] < 2 * TILE_SIZE_SAMPLES
            } ?: return null
            val blockSize = BLOCK_SIZES[index]
            blockProbeAttempts[index]++

            val timing = acquireTimestampBuffers()
            val encoder = device.createCommandEncoder()
            try {
                val opening = GPUPassTimestampWrites(
                    queries,
                    beginningOfPassWriteIndex = 0,
                    endOfPassWriteIndex = Constants.QUERY_SET_INDEX_UNDEFINED
                )
                if (!imageChain.probe(encoder, probed, st.scale, blockSize, opening)) {
                    // Can't run at this size: never try it again.
                    blockProbeAttempts[index] = 2 * TILE_SIZE_SAMPLES
                    encoder.close()
                    releaseTimestampBuffers(timing)
                    continue
                }
                closeTimestampSpan(encoder, queries, timing)
                device.queue.submitAndRelease(encoder)
            } catch (e: Throwable) {
                encoder.close()
                releaseTimestampBuffers(timing)
                throw e
            }
            return measurementScope.launch { measureTileGpuTime(timing, 0, null, 16, 1, blockSize) }
        }
    }

    /** Writes the end of the span a pass opened at query 0, then queues reading both back. */
    private fun closeTimestampSpan(
        encoder: GPUCommandEncoder,
        queries: GPUQuerySet,
        timing: TimestampBuffers
    ) {
        encoder.beginComputePass(
            GPUComputePassDescriptor(
                timestampWrites = GPUPassTimestampWrites(
                    queries,
                    beginningOfPassWriteIndex = Constants.QUERY_SET_INDEX_UNDEFINED,
                    endOfPassWriteIndex = 1
                )
            )
        ).endAndRelease()
        encoder.resolveQuerySet(queries, 0, 2, timing.resolve, 0)
        encoder.copyBufferToBuffer(timing.resolve, 0, timing.result, 0, 16)
    }

    /**
     * Start the generation worker if it isn't running.
     *
     * The worker shares the render thread but not the render mutex, so suspending between
     * batches actually lets a queued frame through - the same reasoning as
     * [WebGpuRenderer.onDispatcher]. A batch ends once its steps' [estimatedNs] reach
     * [BATCH_TARGET_NS], re-read as [tileCostNs] and [blockCostNs] accumulate measurements. Each
     * tile's [generate] job is collected and joined once per batch (not per tile) on `this` coroutine directly - no nested
     * [coroutineScope], and that join is itself what lets a queued frame through, so no separate
     * [yield] is needed. Without [FeatureName.TimestampQuery], [generate] never returns a [Job],
     * so joining an always-empty [measurements] does nothing - pacing falls back to a flat
     * [delay] between [TILES_PER_BATCH_FALLBACK]-sized batches instead.
     */
    private fun schedule() {
        if (workerActive) return
        workerActive = true
        workerScope.launch {
            try {
                while (true) {
                    var spentNs = 0.0
                    var generated = 0
                    var visible = false
                    // Held back, a frame per batch only presents the same fast render again -
                    // still a refresh on e-ink. Grids that got a visible tile, checked once the
                    // batch ends for having nothing visible left to make.
                    val landed = if (progressive) null else HashSet<PageTiles>()
                    val measurements = ArrayList<Job>(MAX_TILES_PER_BATCH)
                    while (generated < MAX_TILES_PER_BATCH && spentNs < BATCH_TARGET_NS) {
                        val req = nextRequest() ?: break
                        try {
                            val started = System.nanoTime()
                            generate(req, this)?.let { step ->
                                step.job?.let { measurements.add(it) }
                                // Plain tiles only: a no-op would drag the average to nothing,
                                // and block encoding isn't per tile.
                                if (step.tile && step.blocks == 0) {
                                    recordTileOverhead((System.nanoTime() - started).toDouble())
                                }
                                spentNs += estimatedNs(step)
                            }
                            generated++
                            // A turn blits prewarmed tiles, so its pages count. Only if it landed:
                            // a full atlas drops it, and a frame would requeue it.
                            val stored = req.state.tiles.containsKey(key(req.tx, req.ty))
                            if (landed == null) {
                                if ((req.onScreen || req.state.page.isOnScreen) && stored) {
                                    visible = true
                                }
                            } else if (req.onScreen && stored) {
                                landed.add(req.state)
                            }
                        } catch (e: CancellationException) {
                            throw e
                        } catch (e: Exception) {
                            Log.e(TAG, "Tile render failed", e)
                        }
                    }
                    if (generated == 0) {
                        // Drained: timing a size we don't use costs nothing here.
                        val probe = try {
                            probeTileSize(this) ?: probeBlockSize(this)
                        } catch (e: CancellationException) {
                            throw e
                        } catch (e: Exception) {
                            Log.e(TAG, "Size probe failed", e)
                            null
                        }
                        (probe ?: break).join()
                        continue
                    }
                    // The frame then finds the grid whole and shows it, or queues what is still
                    // missing - a tile the atlas dropped, say - and the worker carries on.
                    if (landed != null) visible = landed.any { !visiblePending(it) }
                    if (visible) invalidate()
                    if (timestampsSupported) measurements.joinAll() else delay(5.milliseconds)
                }
            } finally {
                workerActive = false
            }
        }
    }

    /** Whether [st] still has a tile queued inside its visible range as of its last draw. */
    private fun visiblePending(st: PageTiles): Boolean = st.pending.any {
        val tx = (it shr 32).toInt()
        val ty = it.toInt()
        tx in st.txMin..st.txMax && ty in st.tyMin..st.tyMax
    }

    /**
     * Pull the highest-priority pending tile: on-screen first, then everything else ([prewarm]'d
     * tiles or a stability-losing grid's leftovers).
     *
     * The on-screen/other split reuses the per-tile distance score: [prewarm] leaves a grid's
     * [PageTiles.txMin]/etc at their empty default, so its tiles always score at or above
     * [OFF_SCREEN_SCORE].
     */
    private fun nextRequest(): Request? {
        var bestPriority = Float.MAX_VALUE
        var bestState: PageTiles? = null
        var bestKey = 0L

        fun priorityOf(st: PageTiles, key: Long): Float {
            val tx = (key shr 32).toInt()
            val ty = key.toInt()
            val centerTx = (st.txMin + st.txMax) * 0.5f
            val centerTy = (st.tyMin + st.tyMax) * 0.5f
            val outX = max(max(st.txMin - tx, tx - st.txMax), 0)
            val outY = max(max(st.tyMin - ty, ty - st.tyMax), 0)
            val cx = tx - centerTx
            val cy = ty - centerTy
            return max(outX, outY) * OFF_SCREEN_SCORE + cx * cx + cy * cy
        }

        val pageIt = pages.iterator()
        while (pageIt.hasNext()) {
            val (_, st) = pageIt.next()
            if (st.pending.isEmpty()) continue
            if (st.destroyed || !st.stable) {
                st.pending.clear()
                continue
            }
            for (pkey in st.pending) {
                val priority = priorityOf(st, pkey)
                if (priority < bestPriority) {
                    bestPriority = priority
                    bestState = st
                    bestKey = pkey
                }
            }
        }

        if (bestState == null) return null
        bestState.pending.remove(bestKey)
        // [priorityOf] adds a whole [OFF_SCREEN_SCORE] per tile outside the grid's on-screen
        // window, so anything at or above it is a tile nobody can currently see - the margin ring
        // kept against a pan, or a grid [prewarm] filled for a page that is not on screen at all.
        return Request(
            bestState, (bestKey shr 32).toInt(), bestKey.toInt(),
            onScreen = bestPriority < OFF_SCREEN_SCORE,
        )
    }

    /**
     * Generates [req], returning the GPU timing measurement's [Job] if this tile started one -
     * launched onto [measurementScope], which [schedule] sets to its own coroutine so a whole
     * batch's worth can be joined together. [blitAvailableTiles]'s direct [generateTileNow] call
     * has no such scope, so it just falls back to [workerScope], fire-and-forget.
     */
    private fun generate(req: Request, measurementScope: CoroutineScope): Generated? {
        val st = req.state
        if (st.destroyed || !st.stable) return null
        if (st.contentVersion != st.page.contentVersion) return null
        return generateTileNow(st, req.tx, req.ty, measurementScope, req.onScreen, blocksPerStep())
    }

    /**
     * What one worker step submitted: a [tile], or only image filter blocks for one. [job] is its
     * timing, null without timestamps.
     */
    private class Generated(val job: Job?, val tile: Boolean, val blocks: Int)

    /**
     * Generate [st]'s tile at ([tx], [ty]) right now if it isn't already cached. Null if nothing
     * was submitted. A tile needing more than [blockLimit] image filter blocks makes only those
     * and is requeued - see [fillBlocks].
     */
    private fun generateTileNow(
        st: PageTiles,
        tx: Int,
        ty: Int,
        measurementScope: CoroutineScope = workerScope,
        onScreen: Boolean = true,
        blockLimit: Int = Int.MAX_VALUE,
    ): Generated? {
        val cpuStart = System.nanoTime()
        val tiles = tilesFilters
        val ts = st.tileSize
        val blockSize = imageChain.blockSize

        // Filters only on screen: a network is a compute dispatch per pass, and the margin ring and
        // [prewarm]'s grids are unseen, so that cost would only compete with the presented frame.
        // They render plain; [drawCore] upgrades them once on screen.

        // Each image's filtered level, its blocks under the tile made here if missing - see
        // [imageLevels] - or else the image itself.
        // Cut already, unless plain and this one is on screen - see [generateTile].
        val existing = st.tiles[key(tx, ty)]
        if (existing != null && !(existing.plain && onScreen)) return null
        val found = if (onScreen) imageLevels(st, tx, ty) else null
        if (found === ImageLevels.Later) return null
        val blank = found === ImageLevels.Blank
        val levels = found as? ImageLevels.Ready
        if (levels != null && levels.missing > blockLimit) {
            fillBlocks(
                st,
                tx,
                ty,
                levels,
                blockLimit,
                blockSize,
                measurementScope,
                true
            )?.let { return it }
        }
        // Unlimited, all now, but [MAX_BLOCKS_PER_STEP] per command buffer.
        if (levels != null && blockLimit == Int.MAX_VALUE) {
            var missing = levels.missing
            while (missing > MAX_BLOCKS_PER_STEP) {
                fillBlocks(
                    st,
                    tx,
                    ty,
                    levels,
                    MAX_BLOCKS_PER_STEP,
                    blockSize,
                    measurementScope,
                    false
                ) ?: break
                val left = missingBlocks(levels)
                if (left >= missing) break
                missing = left
            }
            // Stalled - a filter failed making them. The next frame asks again, after the chain
            // has dropped it.
            if (missing > MAX_BLOCKS_PER_STEP) return null
        }
        val levelOf = levels?.images?.associate { it.image to it.level } ?: emptyMap()
        // Tile pixels per source pixel.
        val sourceScale = if (levels != null) st.scale / levels.k else st.scale

        // Tiles filters' input: the source at its own resolution - the whole source pixels the tile
        // covers, from the first one's start - plus their halo, which [TilesFilterChain.resolve]
        // leaves out.
        val span = ts / sourceScale
        val originX = floor(tx * span)
        val originY = floor(ty * span)
        val inner = tiles.inputSpan(ts, sourceScale)
        val halo = tiles.halo(st.scale)
        val size = inner + 2 * halo
        val runTiles =
            !blank && onScreen && tiles.runsAt(st.scale) && tiles.fits(ts, sourceScale, st.scale)
        // Null: a filter gave up - fall through rather than lose the tile.
        val input = if (runTiles) tiles.input(size, atlas.format, st.scale) else null
        val inputView = tiles.view
        val throughTiles = input != null && inputView != null

        val made = if (levels != null) missingBlocks(levels) else 0
        val mode = TileMode(
            plain = !onScreen,
            opens = made > 0 || throughTiles,
            // The profile's time split covers the tiles filters alone.
            detailed = throughTiles && made == 0,
        )
        val run = if (profile.enabled) TileRun(
            page = profile.pageLabel(st.page),
            tx = tx,
            ty = ty,
            tileSize = ts,
            scale = st.scale,
            onScreen = onScreen,
            filters = if (throughTiles) tiles.names(st.scale) else emptyList(),
            inputSize = if (throughTiles) size else 0,
            imageFilters = if (levels != null || blank) imageChain.names(st.scale) else emptyList(),
            rendered = made,
            blank = blank,
            cpuStart = cpuStart,
        ) else null

        // generateTile skips a tile already cut or without atlas room; a cut one is a new [Tile].
        val before = st.tiles[key(tx, ty)]
        var commits: List<() -> Unit> = emptyList()
        var encoded = 0
        val job = generateTile(
            st, tx, ty, measurementScope, mode, run, { encoded }, blockSize,
            prepare = { encoder, timestamps ->
                // Whichever pass opens first carries the opening timestamp.
                var opening = timestamps
                if (levels != null) {
                    commits = levels.images.map { f ->
                        val filled =
                            imageChain.fill(encoder, f.image, st.scale, f.rect, opening)
                        if (filled.made > 0) opening = null
                        encoded += filled.made
                        filled.commit
                    }
                }
                if (throughTiles) {
                    val pass = encoder.beginRenderPass(clearedColorPass(inputView!!, opening))
                    try {
                        renderTileContent(
                            st,
                            tx,
                            ty,
                            inner,
                            pass,
                            input!!,
                            st.scale / sourceScale,
                            halo.toFloat(),
                            originX,
                            originY,
                            levelOf,
                        )
                    } finally {
                        pass.endAndRelease()
                    }
                    tiles.encode(
                        encoder,
                        st.scale,
                        if (run != null && mode.detailed) timestamps?.querySet else null
                    )
                } else {
                    // [mode] expected a block that wasn't made. Unwritten, the opening timestamp
                    // would be read off stale memory.
                    opening?.let {
                        encoder.beginComputePass(GPUComputePassDescriptor(timestampWrites = it))
                            .endAndRelease()
                    }
                }
            },
        ) { pass, texture ->
            if (!blank) {
                if (throughTiles) {
                    tiles.resolve(
                        pass,
                        atlas.format,
                        sourceScale,
                        st.scale,
                        tx * span - originX,
                        ty * span - originY,
                    )
                } else {
                    renderTileContent(
                        st,
                        tx,
                        ty,
                        ts,
                        pass,
                        texture,
                        st.scale,
                        0f,
                        (tx * ts).toFloat(),
                        (ty * ts).toFloat(),
                        levelOf,
                    )
                }
            }
        }
        // Commit only now: the fill commands are queued, so whatever is queued later sees the blocks.
        // A tile not encoded - already cut, or no atlas room - filled none of them.
        commits.forEach { it() }
        val submitted = st.tiles[key(tx, ty)] !== before
        return if (submitted) Generated(job, true, encoded) else null
    }

    private fun missingBlocks(levels: ImageLevels.Ready) =
        levels.images.sumOf { imageChain.missing(it.image, it.rect) }

    /**
     * Makes up to [limit] of the blocks tile ([tx], [ty]) reads, without cutting it, and requeues
     * the tile, so its blocks spread over several steps. Null if none could be made: the tile is
     * cut as is.
     */
    private fun fillBlocks(
        st: PageTiles,
        tx: Int,
        ty: Int,
        levels: ImageLevels.Ready,
        limit: Int,
        blockSize: Int,
        measurementScope: CoroutineScope,
        requeue: Boolean,
    ): Generated? {
        val queries = timestampQuerySet
        val timing = if (queries != null) acquireTimestampBuffers() else null
        val encoder = device.createCommandEncoder()
        var made = 0
        val commits = ArrayList<() -> Unit>()
        try {
            var opening = queries?.let {
                GPUPassTimestampWrites(
                    it,
                    beginningOfPassWriteIndex = 0,
                    endOfPassWriteIndex = Constants.QUERY_SET_INDEX_UNDEFINED
                )
            }
            for (f in levels.images) {
                if (made >= limit) break
                val filled =
                    imageChain.fill(encoder, f.image, st.scale, f.rect, opening, limit - made)
                if (filled.made > 0) opening = null
                made += filled.made
                commits.add(filled.commit)
            }
            if (made > 0 && queries != null && timing != null) closeTimestampSpan(
                encoder,
                queries,
                timing
            )
            device.queue.submitAndRelease(encoder)
        } catch (e: Throwable) {
            encoder.close()
            timing?.let { releaseTimestampBuffers(it) }
            throw e
        }
        if (made == 0) {
            timing?.let { releaseTimestampBuffers(it) }
            return null
        }
        commits.forEach { it() }
        // Only on progress: blocks a failing filter leaves unmade would loop forever. The next
        // frame asks again.
        if (requeue && missingBlocks(levels) < levels.missing) st.pending.add(key(tx, ty))
        val job = timing?.let {
            measurementScope.launch {
                measureTileGpuTime(
                    it,
                    0,
                    null,
                    16,
                    made,
                    blockSize
                )
            }
        }
        return Generated(job, false, made)
    }

    /**
     * Each image's filtered level for tile ([tx], [ty]), and the part read: the tile, the tiles
     * filters' halo and the resolve's reach. Null where the image filters don't apply at [st]'s
     * scale or a level can't be read whole for this tile - the tile then reads the images.
     * [ImageLevels.Blank] where a level can't be made or held (see [ImageFilterChain.level],
     * [ImageFilterChain.reserve]), logged once per reason; [ImageLevels.Later] where it can, but not
     * this frame.
     */
    private fun imageLevels(st: PageTiles, tx: Int, ty: Int): ImageLevels? = try {
        imageLevelsOrThrow(st, tx, ty)
    } catch (e: Exception) {
        val message = e.message ?: e.toString()
        if (imageChainErrors.add(message)) Log.e(TAG, message)
        ImageLevels.Blank
    }

    private val imageChainErrors = HashSet<String>()

    private fun imageLevelsOrThrow(st: PageTiles, tx: Int, ty: Int): ImageLevels? {
        val chain = imageChain
        if (!chain.isActive) return null
        val geometry = chain.geometry(st.scale) ?: return null
        val k = geometry.k
        val ts = st.tileSize.toFloat()
        val tilesHalo = if (tilesFilters.runsAt(st.scale)) tilesFilters.halo(st.scale) else 0

        val images = ArrayList<FilteredImage>()
        var failed = false
        st.page.forEachImage { image, offsetX, imageScale ->
            val base = image.mipmaps.firstOrNull()
            if (base == null || failed) return@forEachImage
            // Placed as [renderTileContent] places it: centred on the target.
            val si = st.scale * imageScale
            val targetX = -tx * ts + st.scale * offsetX + si * image.offsetX
            val targetY = st.centerYOffset - ty * ts + si * image.offsetY
            val dstLeft = targetX - si * base.width / 2f
            val dstTop = targetY - si * base.height / 2f
            // In image pixels: the tiles filters' halo and the resolve's 2-texel reach are level
            // texels, k per pixel; +1 for where the tile starts.
            val margin = (tilesHalo + 2f) / k + 1f
            val rect = ImageRect(
                -dstLeft / si - margin,
                -dstTop / si - margin,
                (ts - dstLeft) / si + margin,
                (ts - dstTop) / si + margin,
            )
            if (rect.right <= 0f || rect.bottom <= 0f || rect.left >= base.width || rect.top >= base.height) {
                return@forEachImage
            }
            val level = chain.level(image, st.scale)!!
            // The page draw binds a 2x2 window of level tiles, so a tile's read must fit in one.
            // Exceeded only far zoomed out.
            val reads = ts * k / si + 2 * tilesHalo + 2
            if (reads > level.tilesize) {
                failed = true
                return@forEachImage
            }
            images.add(FilteredImage(image, level, rect))
        }
        if (failed || images.isEmpty()) return null
        if (!chain.reserve(
                images.map { ImageFilterChain.Wanted(it.image, it.rect) },
                st.scale,
                frame
            )
        ) {
            return ImageLevels.Later
        }

        val missing = images.sumOf { chain.missing(it.image, it.rect) }
        if (missing > 0 && !chain.ready(st.scale, images[0].level.format)) return null
        return ImageLevels.Ready(k, images, missing)
    }

    /**
     * Render a page's tile, drawing every one of its images into the same pass so a tile
     * straddling their seam comes out with both already in place.
     *
     * Positions each image via [solveImagePlacement], the same inversion of
     * [Image.prepareForRender]'s placement the fast path already uses. The target passed in is
     * this image's ordinary full-frame placement minus the tile's own origin, so solving against
     * a tile-square destination places exactly the crop this tile is responsible for.
     */
    private fun renderTileContent(
        st: PageTiles,
        tx: Int,
        ty: Int,
        tileSize: Int,
        pass: GPURenderPassEncoder,
        texture: GPUTexture,
        scale: Float = st.scale,
        inset: Float = 0f,
        originX: Float = tx * tileSize.toFloat(),
        originY: Float = ty * tileSize.toFloat(),
        /** Each image's filtered level, to draw from instead of the image's own. */
        levels: Map<Image, Mipmap>? = null,
    ) {
        // [tileSize] is in this destination's pixels, so it already carries [scale];
        // [centerYOffset] is in the grid's and has to be brought across. [inset] widens the
        // destination without moving the tile within it.
        val ts = tileSize.toFloat()
        val s = scale
        val centerY = st.centerYOffset * (scale / st.scale)
        val dst = ts + 2f * inset
        // With filters on, every tile reads the base level, filtered or not, so a plain fallback
        // matches its replacement.
        val levelZero = imageChain.isActive || tilesFilters.isActive

        // A halo past the page's edge would read nothing, which a network takes for black, darkening
        // the outermost rows. So the edges are extended as a clamped read would: top and bottom for
        // every image, sides only at the outer ends - between two pages, the other page is there.
        val images = ArrayList<Triple<Image, Float, Float>>()
        st.page.forEachImage { image, offsetX, imageScale ->
            if (image.mipmaps.isNotEmpty()) images.add(Triple(image, offsetX, imageScale))
        }
        if (images.isEmpty()) return
        val leftmost = images.minOf { it.second }
        val rightmost = images.maxOf { it.second }

        Hdr.cuttingTiles = true
        try {
            for ((image, srcOffsetX, imageScale) in images) {
                val si = s * imageScale
                val targetX = -originX + inset + s * srcOffsetX + si * image.offsetX
                val targetY = centerY - originY + inset + si * image.offsetY
                val (x, y) = solveImagePlacement(targetX, targetY, si, image, dst, dst)
                val extend = if (inset > 0f) Extend(
                    left = if (srcOffsetX == leftmost) inset else 0f,
                    top = inset,
                    right = if (srcOffsetX == rightmost) inset else 0f,
                    bottom = inset,
                ) else null
                tilesFilters.draw(
                    pass,
                    image,
                    texture,
                    x,
                    y,
                    si,
                    levels?.get(image),
                    levelZero,
                    extend
                )
            }
        } finally {
            Hdr.cuttingTiles = false
        }
    }

    /**
     * Compact, stable-per-object identifier for [page] in log messages - "current"/"next" are
     * relative labels that get reassigned to a different actual page on every turn, so this is
     * what lets a log reader tell whether two log lines are really about the same page.
     */
    private fun pageId(page: ImagePage.ImageSingle) =
        Integer.toHexString(System.identityHashCode(page))

    /**
     * Shared setup for [renderFullyTiled]/[blitAvailableTiles]: draws whatever's cached into
     * [dst] and queues anything missing (via [drawCore]'s own `schedule()` call), returning the
     * grid's state - or null if [page] has no drawable images.
     */
    private fun drawGridForFullPage(
        pass: GPURenderPassEncoder, page: ImagePage.ImageSingle, dst: GPUTexture
    ): PageTiles? {
        if (page.destroyed || !page.highQuality) return null
        // This path draws cached tiles without touching the image, so it has to report for itself.
        page.currentImage?.let { if (it.isHdr) Hdr.noteHdrDrawn(it, it.hdrHeadroom) }
        if (!page.hasUploadedImage) return null

        val a = pagedAnchor(page, dst, 0f, 0f, 1f)
        drawCore(
            pass,
            page,
            dst,
            a.anchorX,
            a.anchorY,
            0f,
            a.pageScale,
            suppressGeneration = false,
            applyRetainWindow = false
        )

        val st = pages[page] ?: return null

        // A scale (or centerYOffset) change wipes the grid and marks it unstable within that same
        // drawCore call - which also means its want/pending pass ran before stability was granted,
        // so nothing got queued. drawCore's two-call gate exists to avoid re-wiping every frame
        // while a gesture actively drives scale, but this caller has no next call to benefit from
        // that - it must finish now regardless, e.g. a page turn triggered while the page is still
        // mid-animation back to its home scale. Re-running once more (same anchor/scale, so no
        // further wipe) grants stability immediately and lets the want/pending pass actually queue
        // this frame's tiles.
        if (!st.stable) {
            drawCore(
                pass,
                page,
                dst,
                a.anchorX,
                a.anchorY,
                0f,
                a.pageScale,
                suppressGeneration = false,
                applyRetainWindow = false
            )
        }

        return st
    }

    /**
     * Render [page]'s full tile grid into [dst], generating any tile the worker hasn't reached
     * yet right here rather than leaving it queued - for callers that can't show less than fully
     * complete. [blitAvailableTiles] is the partial/progressive counterpart. Returns false,
     * drawing nothing, if the page has no drawable images.
     */
    fun renderFullyTiled(
        pass: GPURenderPassEncoder, page: ImagePage.ImageSingle, dst: GPUTexture
    ): Boolean {
        val st = drawGridForFullPage(pass, page, dst) ?: return false
        if (st.pending.isEmpty()) return true

        // Missing tiles only: a plain one awaiting its re-cut is already drawn, and the worker replaces it.
        val toGenerate = st.pending.filter { !st.tiles.containsKey(it) }
        st.pending.removeAll(toGenerate.toSet())
        toGenerate.forEach { tkey -> generateTileNow(st, (tkey shr 32).toInt(), tkey.toInt()) }

        // Only what just landed: premultiplied-over, so drawing a tile twice differs from once.
        drawTiles(pass, dst.format, st, toGenerate)
        return true
    }

    /**
     * Blit whatever's already cached into [dst], queuing anything missing for the background
     * worker rather than force-generating it - see [renderFullyTiled] for the force-complete
     * counterpart. Used by [Transition]'s cache to layer tiles over an immediate fast-rendered
     * seed without blocking on full coverage. Returns false, drawing nothing, if [page] has no
     * drawable images.
     */
    fun blitAvailableTiles(
        pass: GPURenderPassEncoder, page: ImagePage.ImageSingle, dst: GPUTexture
    ): Boolean = drawGridForFullPage(pass, page, dst) != null

    /**
     * Render one tile at ([tx], [ty]) into [st] via [render], then store it. The GPU
     * timing measurement this starts is launched onto [measurementScope] - see [generate]'s doc
     * for why that's a per-batch scope from [schedule] rather than [workerScope] directly.
     */
    private inline fun generateTile(
        st: PageTiles,
        tx: Int,
        ty: Int,
        measurementScope: CoroutineScope,
        mode: TileMode,
        run: TileRun?,
        blocks: () -> Int,
        blockSize: Int,
        prepare: (GPUCommandEncoder, GPUPassTimestampWrites?) -> Unit,
        render: (GPURenderPassEncoder, GPUTexture) -> Unit
    ): Job? {
        val key = key(tx, ty)
        // Cut already, unless plain and this cut isn't.
        val existing = st.tiles[key]
        if (existing != null && !(existing.plain && !mode.plain)) return null
        evict()

        val pool = atlas
        // Slabs belong to one size at a time, so a just-changed size can find them all spoken
        // for while the budget says there is room - free a grid and try once more.
        var origin = pool.acquire(st.tileSize)
        if (origin < 0) {
            freeColdestGrid(st)
            origin = pool.acquire(st.tileSize)
        }
        // Still nothing - the worker comes back to this tile.
        if (origin < 0) return null

        try {
            return encodeTile(
                st,
                key,
                origin,
                pool,
                measurementScope,
                mode,
                run,
                blocks,
                blockSize,
                prepare,
                render
            )
        } catch (e: Throwable) {
            pool.release(st.tileSize, origin)
            throw e
        }
    }

    private inline fun encodeTile(
        st: PageTiles,
        key: Long,
        origin: Int,
        pool: TileAtlas,
        measurementScope: CoroutineScope,
        mode: TileMode,
        run: TileRun?,
        blocks: () -> Int,
        blockSize: Int,
        prepare: (GPUCommandEncoder, GPUPassTimestampWrites?) -> Unit,
        render: (GPURenderPassEncoder, GPUTexture) -> Unit
    ): Job? {
        val queries = timestampQuerySet
        if (queries == null) {
            val encoder = device.createCommandEncoder()
            // Before the pass opens: a compute pass cannot nest inside a render pass.
            prepare(encoder, null)
            val pass = encoder.beginRenderPass(clearedColorPass(pool.scratchView(st.tileSize)))
            try {
                render(pass, pool.scratch(st.tileSize))
            } finally {
                pass.endAndRelease()
            }
            pool.copyScratchInto(encoder, origin, st.tileSize)
            device.queue.submitAndRelease(encoder)
            storeTile(st, key, origin, mode)
            if (run != null) {
                run.cpuMs = (System.nanoTime() - run.cpuStart) / 1e6
                profile.report(run, null)
            }
            return null
        }

        val timing = acquireTimestampBuffers()
        val encoder = device.createCommandEncoder()
        // A detailed profiled tile also writes the page draw's end, each filter's pass (see
        // [TilesFilterChain.encode]) and the final resize's start.
        val detailed = run != null && mode.detailed
        val count = if (detailed) PROFILE_QUERY_COUNT else 2
        try {
            // [prepare]'s passes run before this one and would otherwise go unmeasured, and
            // [schedule] paces batches by this number - a tile that reads as free gets queued
            // eight at a time. So the first pass [prepare] opens carries the opening
            // timestamp, leaving only the closing one here.
            //
            // Per tile, not per renderer: a tile the chain declines, or one that reads every block
            // from the cache, has no first pass to carry the opening write. Getting that wrong
            // leaves query 0 unwritten and the elapsed time read off stale memory.
            val opening = if (mode.opens) {
                prepare(
                    encoder, GPUPassTimestampWrites(
                        queries,
                        beginningOfPassWriteIndex = 0,
                        endOfPassWriteIndex = if (detailed) 2 else Constants.QUERY_SET_INDEX_UNDEFINED
                    )
                )
                if (detailed) RESIZE_BEGIN_QUERY else Constants.QUERY_SET_INDEX_UNDEFINED
            } else {
                prepare(encoder, null)
                0
            }

            val pass = encoder.beginRenderPass(
                clearedColorPass(
                    pool.scratchView(st.tileSize), timestampWrites = GPUPassTimestampWrites(
                        queries, beginningOfPassWriteIndex = opening, endOfPassWriteIndex = 1
                    )
                )
            )
            try {
                render(pass, pool.scratch(st.tileSize))
            } finally {
                pass.endAndRelease()
            }

            pool.copyScratchInto(encoder, origin, st.tileSize)
            encoder.resolveQuerySet(queries, 0, count, timing.resolve, 0)
            encoder.copyBufferToBuffer(timing.resolve, 0, timing.result, 0, count * 8L)

            device.queue.submitAndRelease(encoder)
            if (run != null) run.cpuMs = (System.nanoTime() - run.cpuStart) / 1e6
        } catch (e: Throwable) {
            encoder.close()
            releaseTimestampBuffers(timing)
            throw e
        }
        storeTile(st, key, origin, mode)

        val made = blocks()
        return measurementScope.launch {
            measureTileGpuTime(
                timing,
                st.tileSize,
                run,
                count * 8,
                made,
                blockSize
            )
        }
    }

    /** Keep a just-cut tile, retiring the plain one it replaces - see [retired]. */
    private fun storeTile(st: PageTiles, key: Long, origin: Int, mode: TileMode) {
        st.tiles[key]?.let { retired.add(st.tileSize to it.atlasOrigin) }
        st.tiles[key] = Tile(origin).also { it.lastUsed = frame; it.plain = mode.plain }
        st.instancesDirty = true
    }

    /** [blocks] of [blockSize] were made in the span; a [tileSize] of 0 means nothing else was. */
    private suspend fun measureTileGpuTime(
        timing: TimestampBuffers,
        tileSize: Int,
        run: TileRun?,
        bytes: Int,
        blocks: Int,
        blockSize: Int,
    ) {
        val result = timing.result
        try {
            awaitPumped { result.mapAndAwait(MapMode.Read, 0, bytes.toLong()) }
        } catch (e: CancellationException) {
            // Still in flight, possibly - not safe to hand back.
            timing.resolve.destroyAndRelease()
            result.destroyAndRelease()
            throw e
        } catch (e: Exception) {
            // A lost device fails every pending map. The timing only paces tile batches, and
            // rethrowing escaped the worker scope as an uncaught exception that killed the app.
            timing.resolve.destroyAndRelease()
            result.destroyAndRelease()
            Log.w(TAG, "Tile timing unavailable: ${e.message}")
            return
        }
        val mapped = result.getConstMappedRange(0, bytes.toLong())
        mapped.order(ByteOrder.nativeOrder())
        val timestamps = LongArray(bytes / 8) { mapped.getLong(it * 8) }
        result.unmap()
        releaseTimestampBuffers(timing)
        val start = timestamps[0]
        val end = timestamps[1]
        if (start > 0 && end > start) {
            val elapsed = (end - start).toDouble()
            if (blocks == 0) {
                recordTileCost(tileSize, elapsed)
            } else if (tileSize == 0) {
                recordBlockCost(blockSize, elapsed / blocks)
            } else {
                // A tile and its blocks: split only once the tile alone has been measured.
                val tile = tileCostNs.getOrElse(sizeIndex(tileSize)) { 0.0 }
                if (tile > 0.0) recordBlockCost(
                    blockSize,
                    (elapsed - tile).coerceAtLeast(0.0) / blocks
                )
            }
        }
        if (run != null) profile.report(run, timestamps)
    }

    private suspend fun awaitPumped(block: suspend () -> Unit) = coroutineScope {
        val pump = launch {
            while (isActive) {
                WebGpuRenderer.instance.processEvents()
                delay(1.milliseconds)
            }
        }
        try {
            block()
        } finally {
            pump.cancel()
        }
    }

    private fun clearedColorPass(
        view: GPUTextureView, timestampWrites: GPUPassTimestampWrites? = null
    ) = GPURenderPassDescriptor(
        colorAttachments = arrayOf(
            GPURenderPassColorAttachment(
                view = view,
                loadOp = LoadOp.Clear,
                storeOp = StoreOp.Store,
                clearValue = GPUColor(0.0, 0.0, 0.0, 0.0)
            )
        ), timestampWrites = timestampWrites
    )

    /** Where [st]'s next blits land in [dst]: its snapped anchor and clip rect, in pixels. */
    private fun placeGrid(
        st: PageTiles,
        dst: GPUTexture,
        snapX: Float,
        snapY: Float,
        clipL: Float,
        clipT: Float,
        clipR: Float,
        clipB: Float
    ) {
        st.snapX = snapX
        st.snapY = snapY
        st.dstW = dst.width.toFloat()
        st.dstH = dst.height.toFloat()
        st.clipL = clipL
        st.clipT = clipT
        st.clipR = clipR
        st.clipB = clipB
    }

    /** What the budgeted tiles come to - the area every tile size is held to, whatever its own. */
    private val maxTileBytes
        get() = (if (atlasOrNull != null) atlasBudgetTiles else budgetTiles()).toLong() * TILE_BYTES

    private fun tileBytes(st: PageTiles) = st.tileSize.toLong() * st.tileSize * 4

    /**
     * Evict least-recently-used tiles down to [maxTileBytes], best-effort.
     *
     * Tiles used this frame or last are never touched: a freed slot is handed straight back out,
     * and rewriting one a recorded pass reads would draw the wrong content. If everything is that
     * fresh the cap overshoots, bounded like the wanted set by the viewport.
     */
    private fun evict() {
        var total = 0L
        for (st in pages.values) total += st.tiles.size * tileBytes(st)
        if (total < maxTileBytes) return

        val candidates = ArrayList<Triple<PageTiles, Long, Tile>>()
        for (st in pages.values) {
            for ((k, t) in st.tiles) if (t.lastUsed < frame - 1) candidates.add(Triple(st, k, t))
        }
        candidates.sortBy { it.third.lastUsed }
        var i = 0
        while (total >= maxTileBytes && i < candidates.size) {
            val (st, k, t) = candidates[i]
            st.tiles.remove(k)
            st.instancesDirty = true
            atlasOrNull?.release(st.tileSize, t.atlasOrigin)
            total -= tileBytes(st)
            i++
        }
    }

    /**
     * Free every tile. Safe from any thread; destruction runs on the render thread. Not a
     * permanent shutdown - the surface can be recreated with the same viewer state afterward, and
     * rendering simply refills the cache.
     */
    fun cleanup() {
        workerScope.launch {
            // Nothing to free on a lost device, and calling into its objects crashed in native code.
            if (!WebGpuRenderer.isAvailable) {
                pages.clear()
                atlasOrNull = null
                retired.clear()
                timestampPool.clear()
                stencilTextures.fill(null)
                stencilViews.fill(null)
                stencilWidth = 0
                stencilHeight = 0
                lutBindGroup = null
                if (RenderPage.lut === imageLut) RenderPage.lut = null
                return@launch
            }
            // On the worker with everything else it owns: the chains' textures can be mid-tile
            // when the view is torn down. Under the mutex, so no frame is recording with it.
            WebGpuRenderer.withContext {
                imageChain.cleanup()
                tilesFilters.cleanup()
                lutBindGroup?.close()
                lutBindGroup = null
                imageLut.cleanup()
                if (RenderPage.lut === imageLut) RenderPage.lut = null
                pages.values.forEach { it.destroyAll(atlasOrNull) }
                pages.clear()
                atlasOrNull?.destroy()
                atlasOrNull = null
                retired.clear()
                timestampPool.forEach { it.resolve.destroyAndRelease(); it.result.destroyAndRelease() }
                timestampPool.clear()
                for (i in 0 until STENCIL_BUFFER_COUNT) {
                    stencilViews[i]?.close()
                    stencilViews[i] = null
                    stencilTextures[i]?.destroyAndRelease()
                    stencilTextures[i] = null
                }
                stencilWidth = 0
                stencilHeight = 0
            }
        }
    }
}

private const val BLIT_SHADER = """
struct FrameParams {
    // Snapped screen-pixel position of the grid's centre; the tile grid hangs off it.
    snap: vec2<f32>,
    dst_size: vec2<f32>,
    // The grid's rect in screen pixels that blits are clipped to - see the class doc.
    clip: vec4<f32>,
    // This grid's tile size and the atlas's - per grid, so neither can be a constant.
    ts: f32,
    atlas_size: f32,
}

var<immediate> frame_params: FrameParams;
@group(0) @binding(1) var src_tex: texture_2d<f32>;
@group(0) @binding(2) var src_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Per instance: grid coordinate, then position in the atlas.
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32, @location(0) tile: vec4<f32>) -> VertexOutput {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), // Top-left
        vec2<f32>(0.0, 1.0), // Bottom-left
        vec2<f32>(1.0, 0.0), // Top-right
        vec2<f32>(1.0, 0.0), // Top-right
        vec2<f32>(0.0, 1.0), // Bottom-left
        vec2<f32>(1.0, 1.0)  // Bottom-right
    );

    let pos = corners[vertex_index];
    let ts = frame_params.ts;

    // Clamping the corners to the clip rect shrinks the quad and its uv window in step, so
    // whatever survives is still a 1:1 texel copy. Offscreen overhang is left to NDC clipping.
    let origin = frame_params.snap + tile.xy * ts;
    let p = clamp(origin + pos * ts, frame_params.clip.xy, frame_params.clip.zw);

    var out: VertexOutput;
    out.position = vec4<f32>(
        p.x / frame_params.dst_size.x * 2.0 - 1.0,
        1.0 - p.y / frame_params.dst_size.y * 2.0,
        0.0, 1.0
    );
    // Into the atlas: this tile's slot plus whatever of the tile survived the clamp.
    out.uv = (tile.zw + (p - origin)) / frame_params.atlas_size;
    return out;
}

"""

private const val BLIT_FS = """
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // 1:1 at integer positions with a nearest sampler: an exact copy of the tile's texels,
    // already premultiplied by RenderPage.
    return textureSample(src_tex, src_sampler, in.uv);
}
"""

private const val BLIT_LUT_FS = """
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let texel = textureSample(src_tex, src_sampler, in.uv);
    if (texel.a <= 0.0) {
        return texel;
    }
    return vec4<f32>(apply_lut(texel.rgb / texel.a) * texel.a, texel.a);
}
"""
