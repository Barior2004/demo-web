# Benchmark measurement protocol v1

Each completed image/video run is saved once to SQLite with a UUID, UTC times,
model/input SHA256, requested inference settings, device, Python/package versions,
available ONNX providers, configured session providers, metrics and their definitions.
No image/video bytes or detection coordinates are stored in the database.

## Timing boundaries

- Model call: wall time around `predict`, including a CUDA synchronization after
  prediction on GPU. A preceding synchronization and the shared-model lock wait
  are excluded from model call time. This is not GPU kernel time.
- Internal preprocess/inference/postprocess: reported by Ultralytics, retained
  separately. Their sum need not match model call time.
- Model FPS: number of samples / sum of model call seconds. Images repeat the
  same image; videos use consecutive frames. These workloads are different.
- p50/p95: NumPy linear percentiles of model call milliseconds.
- Active video processing: measured per frame from before decode through
  inference, drawing, encoder submission and any preview color conversion.
  Includes lock waits. Excludes Streamlit submission, metrics UI, pacing and
  list bookkeeping. Final encoder close/flush time is added at completion.
  This is a scoped measurement during paced playback, not a claim about maximum
  throughput, browser FPS or complete surveillance system performance.
- Server wall FPS: processed frames / wall seconds from loop start through encoder
  close. Includes preview calls, UI, pacing, bookkeeping and final EOF read.
  Excludes upload, hashing, opening input/encoder, database writes, serving the
  completed MP4 and browser rendering/network delivery acknowledgements.
- Preview submission FPS: number of `st.image` submissions / the same server wall
  seconds. Sending a frame does not prove that the browser displayed it.
- Encode submission milliseconds measure buffered writes, not complete encoding.
  Finalization time is saved separately and included in final active/wall FPS.
- Image benchmark wall time includes one color conversion, repeated prediction,
  progress updates and one final drawing. It excludes setup and database writes.

Live video metrics are provisional. After the last frame, final statistics are
recomputed from all frames and include encoder finalization, even when the last
frame was not selected for preview. Those final statistics are saved/displayed.
Zero-duration samples produce null FPS instead of infinity.
The live statistics panel updates at most twice per second to reduce UI overhead;
this does not reduce inference/frame recording or configured preview submissions.

Warm-up is done on synthetic frames when the shared model is loaded, not for every
click. The first real image/video may still incur shape-dependent initialization.
The model cache key includes its SHA256. Concurrent predict calls on the same
cached model are serialized; GPU competition from other processes is not controlled.
Use an otherwise idle instance and fixed inputs/settings for comparisons.

## Persistence

Local default: `data/benchmarks.db` relative to the application working directory.
Override with `BENCHMARK_DB_PATH`. GPU Compose mounts `./data` to `/app/data` and
uses `/app/data/benchmarks.db`. Recreating the container preserves this directory.
It is not an off-instance backup: deleting the EC2 disk/directory loses the data.
Do not commit databases or their journal files. Storage errors are shown explicitly;
a successful analysis with a failed save is never shown as a successful save.
`APP_REVISION` can be supplied by a future deployment step; until then the stored
app revision is explicitly `unknown`. The measurement protocol is always versioned.

Run tests using `python -m unittest discover -s tests -v`. The storage/formula tests
do not need a model or GPU. The optional Streamlit AppTest checks use a mocked
predictor and real short video encoding; they verify the completed-run save path,
rerun behavior and error messages, not model accuracy or GPU performance.

## Local acceptance

Run an image and a short video. Each should show a distinct saved run ID.
Rerunning the Streamlit UI without clicking Analyze must not insert another row.
Restart the application and query SQLite to confirm both rows remain. Final video
count must match the processed count, including a video with frame count not
divisible by the preview interval. Do not compare the renamed metrics directly
with historical Compute/Realtime Preview FPS values without their old definitions.
