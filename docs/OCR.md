# Local OCR evidence

OCR detects and recognizes sampled visible text using local ONNX Runtime CPU inference. It saves raw readings, boxes, model provenance, actual decoded timestamps and evidence JPEGs. Text is untrusted content; it never becomes instructions, captions, a corrected transcript, or clip-selection ground truth. No OCR job downloads weights or contacts a paid service.

## Options and defaults

The defaults remain `ocr_model=ppocr-v4`, `ocr_mode=baseline` and `ocr_quality_samples=1`. `ocr_sample_seconds` accepts 1–600 seconds (default 5); `ocr_max_samples` accepts 2–120 (default 120). Sampling spans the detected video extent, increasing its spacing on longer videos within the sample cap. Sources must be at most 12 hours long. The CLI exposes `ocr PROJECT_ID --model ... --mode ... --quality-samples ...`; the UI and API expose the same model, mode and quality settings.

- **Baseline:** preserves the existing 960-pixel longest-edge detector view and at most 50 lines per frame. Boxes tilted 45 degrees or more from horizontal are skipped; this geometric test can also skip upright narrow glyphs.
- **Detail:** retains the original resolution within a 4,096-pixel edge and 8,847,360-pixel area, otherwise recording the reduction. A reduced global pass plus up to 20 sequential 960-pixel tiles can recover some small text. Tiles overlap by at least 128 pixels; overlap can be larger at frame edges. Boxes map back to the saved bounded frame. Raw readings, including overlapping results, are preserved; matching accepted overlaps are linked. Tall or steep regions use an explicitly labeled axis-aligned, unrotated fallback. This can recover upright narrow digits but does not establish reading orientation.
- **Quality sampling:** requests up to one, two or three candidate frames. Two candidates request the center and 0.25 seconds earlier; three add 0.25 seconds later. Requests are clipped to the sampling extent. When comparing frames, bounded grayscale Laplacian variance selects the sharpest available candidate, and only that frame receives OCR. Actual selected PTS, candidate sharpness and evidence paths are retained. Decode fallbacks can use an earlier frame, whose actual timestamp is also recorded. Sharpness measures texture, not text correctness.
- **Temporal agreement:** detail mode links exact normalized text in adjacent sampled frames when their axis-aligned boxes have intersection-over-union of at least 0.5. It retains raw strings and one-frame observations, and caps the report at 500 links. Agreement supplies support, not a corrected reading or evidence of correctness.

Inference uses two CPU threads and batch size one. The 1.5 GiB available-RAM guard remains. Detail mode adds a cooperative 1.25 GiB process-tree RSS check and reserves another 128 MiB plus 16 bytes per source pixel above the protected RAM margin. It rejects source geometry above 40 million pixels or a 32:1 aspect ratio, and limits each saved JPEG to 24 MiB. Recognition crops remain 48 pixels high and at most 960 pixels wide.

Detail permits at most 200 raw lines per frame, versus 50 in baseline. Both modes cap retained evidence at 6,000 raw lines per job, 2 MiB of raw text, an 8 MiB serialized report and 512 MiB of saved frames. Sample records have a separate 7 MiB budget within the report limit. Resource and 600-second wall-time checks run between native calls; they do not continuously interrupt a running ONNX call. FFmpeg decoding has a 20-second timeout, and worker process cancellation remains available.

Successful reports flag truncated readings and unprocessed samples or passes. Exceeding RAM, time or frame-storage limits fails the job; it does not guarantee a complete partial report. API job lists omit OCR result bytes in SQL before loading rows, while job detail and the latest-project OCR endpoint retain full evidence.

## Verified model identities

Default **PP-OCRv4**, `SWHL/RapidOCR`, revision `4e4644045a07c403b1ad40ca97340ecb4a8dc2c1`, Apache-2.0:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `PP-OCRv4/ch_PP-OCRv4_det_infer.onnx` | 4,745,517 | `d2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9` |
| `PP-OCRv4/ch_PP-OCRv4_rec_infer.onnx` | 10,857,958 | `48fc40f24f6d2a207a2b1091d3437eb3cc3eb6b676dc3ef9c37384005483683b` |

Optional experimental **PP-OCRv5 mobile**, RapidAI conversion of PaddlePaddle models, Apache-2.0:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `ch_PP-OCRv5_det_mobile.onnx` | 4,819,576 | `4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae` |
| `ch_PP-OCRv5_rec_mobile.onnx` | 16,631,306 | `5825fc7ebf84ae7a412be049820b4d86d77620f204a041697b0494669b1742c5` |

The 21,450,882-byte pair was explicitly downloaded and verified, then loaded by the existing ONNX Runtime 1.23.2 installation; no software package was installed. Catalog revision `bd941dae09da793e76c6ba15c7aaf40f8424232a` identifies the RapidOCR source catalog, not an immutable ModelScope weight revision. Hosting uses the version path `v3.10.0`; exact byte sizes and SHA-256 hashes enforce weight identity. These are RapidAI ONNX conversions of official Paddle models. [Official detector](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det/tree/0d63e78e2b680928f6b1747d76a08db6e645efb7), [official recognizer](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec/tree/682f20538d8c086cb2128e5cfac775e6c4904e85), [pinned RapidOCR catalog](https://github.com/RapidAI/RapidOCR/blob/bd941dae09da793e76c6ba15c7aaf40f8424232a/python/rapidocr/default_models.yaml).

Each embedded dictionary is validated against its model-specific output dimension. V4 uses 6,623 dictionary tokens plus CTC blank and space. V5 uses 18,383 unique tokens plus blank and space, and verifies dictionary SHA-256 `17665d27ed39f0deb82007859992d626d3105d0ee4578c120b7c72138dc04d05`. V5 tokens contain at most two codepoints and eight UTF-8 bytes; ten are two-codepoint flag tokens. A different dictionary is not silently substituted.

Both models use the same BGR preprocessing, 48-pixel recognizer height, 0.5 detection-box acceptance threshold and expanded rotated rectangle factor of 1.6 for comparison. This bounded implementation differs from upstream polygon unclip and the official V5 detector settings of 0.6 box acceptance and 1.5 unclip ratio. It omits an orientation classifier and makes no claim of upstream performance parity. See [retained notices](licenses/ocr/NOTICE.txt).

## Measured quality and limits

OCR performance needs validation on each intended content type. Exact frame bytes, labels and scoring were frozen before new outputs. No thresholds or labels were tuned after the comparison. Detail increased selected-label recall and also produced extra wrong or partial readings. V5 improved spacing on these controls, but did not improve every content category. Both modes require review; the defaults remain unchanged.

Chinese/English models do not establish general multilingual OCR. Small, curved, crowded, rotated, upside-down and handwritten text may be wrong or absent. Model scores are uncalibrated. A partial label set cannot establish precision or false positives outside labeled regions. These small diagnostic sets do not establish broad accuracy or throughput.

The previous two-frame basketball baseline matched 1/5 selected labels; its raw evidence and annotations are retained. Earlier geometric skipping could omit upright narrow digits. New detail-mode results and regression coverage are reported separately, preserving that historical outcome.
