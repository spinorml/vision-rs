# Training

Training support lives behind the `training` feature (on by default) and,
for the GPU loss kernels, the `cuda` feature. `vision_rs::models::yolo::loss`
implements target assignment and the loss functions needed to train YOLO26
from raw model outputs against ground-truth boxes.

## Anchor grid

`AnchorGrid` (in `loss::anchor`) precomputes anchor centres for all three
FPN levels (strides 8/16/32), flattened into one array ordered level-by-level
then row-major within each level. `AnchorGrid::yolo26(img_h, img_w)` builds
the grid for a given input resolution.

## Target assignment

`TaskAlignedAssigner` (in `loss::assign`) is a simplified, CPU-side
implementation of ultralytics' TaskAlignedAssigner. For each ground-truth
box, it scores every anchor as:

```text
score = cls_score^alpha * iou^beta
```

and assigns the top-`k` anchors per GT as positives (conflicts — multiple
GTs claiming the same anchor — are broken by highest score). The result
(`AssignResult`) carries, per anchor: whether it's positive, the assigned
GT box/class, and a *soft target* — `(align / max_align_for_gt) *
max_iou_for_gt` — used as both the soft classification label and the box
loss weight, matching ultralytics' E2ELoss normalisation.

## `Yolo26Loss` (CUDA)

`Yolo26Loss::new(img_h, img_w, nc, cap)` builds the loss state: the anchor
grid, a default assigner (`top_k = 10`) for the one2many head, and a
`top_k = 1` assigner for the one2one head.

```rust,ignore
pub fn compute_grads(
    &self, device: &CudaDevice<'_>,
    boxes: &[f32], scores: &[f32],
    gt_boxes_b: &[Vec<[f32; 4]>], gt_cls_b: &[Vec<usize>],
) -> anyhow::Result<(Vec<f32>, Vec<f32>)>;
```

Compiles and runs the CIoU and classification-loss forward/backward kernels
(see [Custom Kernels](../kernels-and-performance/custom-kernels.md)) for a
single batch, returning `(d_boxes, d_scores)` gradients ready to backprop
into the model graph.

### Dual-head training

```rust,ignore
pub fn compute_grads_dual(
    &self, device: &CudaDevice<'_>,
    boxes_o2m: &[f32], scores_o2m: &[f32],
    boxes_o2o: &[f32], scores_o2o: &[f32],
    gt_boxes_b: &[Vec<[f32; 4]>], gt_cls_b: &[Vec<usize>],
    w_o2m: f32, w_o2o: f32,
) -> anyhow::Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)>;
```

Runs TAL assignment independently for both heads (their own assigners),
scaling the resulting gradients by `w_o2m`/`w_o2o`. Pair this with
[`yolo26_dual`](./yolo26-architecture.md) for the matching dual-head forward
pass.

**Loss weight schedule** (ultralytics-style): `w_o2m = 1.0` constant
throughout training; `w_o2o = step / total_steps`, ramping 0→1 linearly so
the one2one head — the one actually used at inference — gradually takes
over by the end of training. The caller controls the schedule; `Yolo26Loss`
just applies the weights you pass in.

> One2many is traced *before* one2one in `yolo26_dual`'s forward closure —
> the training loop relies on this ordering for stable DAG node
> identification. If you're writing a custom training loop against the
> traced graph directly, don't reorder the two head calls.

## The `yolo26` example's `Train`/`DebugTrain` subcommands

`examples/yolo26.rs` has a full CLI training loop wired up against this
API — see its `Train` subcommand for a working reference implementation,
and `DebugTrain` for a variant that dumps intermediate gradient statistics
(useful when debugging a new loss/kernel change).
