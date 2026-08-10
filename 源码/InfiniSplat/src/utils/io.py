from pathlib import Path
from typing import Union

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
import torch
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from torch import Tensor

FloatImage = Union[
    Float[Tensor, "height width"],
    Float[Tensor, "channel height width"],
    Float[Tensor, "batch channel height width"],
]


def prep_image(image: FloatImage) -> UInt8[np.ndarray, "height width channel"]:
    # Handle batched images.
    if image.ndim == 4:
        image = rearrange(image, "b c h w -> c h (b w)")

    # Handle single-channel images.
    if image.ndim == 2:
        image = rearrange(image, "h w -> () h w")

    # Ensure that there are 3 or 4 channels.
    channel, _, _ = image.shape
    if channel == 1:
        image = repeat(image, "() h w -> c h w", c=3)
    assert image.shape[0] in (3, 4)

    image = (image.detach().clip(min=0, max=1) * 255).type(torch.uint8)
    return rearrange(image, "c h w -> h w c").cpu().numpy()


def save_video(
    images: list[FloatImage],
    path: Union[Path, str],
    fps: int = 10,
) -> None:
    """Save an RGB video whose input frames are in range 0-1."""

    # Create the parent directory if it doesn't already exist.
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    frames = [prep_image(image) for image in images]

    if len(frames) == 0:
        raise ValueError("save_video received an empty frame list.")

    reference_height, reference_width = frames[0].shape[:2]
    reference_channels = frames[0].shape[2] if frames[0].ndim == 3 else 1
    normalized_frames = []
    for idx, frame in enumerate(frames):
        if frame.ndim != 3:
            raise ValueError(f"Frame {idx} has invalid shape {frame.shape}; expected HWC.")
        if frame.shape[2] != reference_channels:
            raise ValueError(
                f"Frame {idx} has channel count {frame.shape[2]} but expected {reference_channels}."
            )
        if frame.shape[:2] != (reference_height, reference_width):
            raise ValueError(
                f"Frame {idx} has shape {frame.shape[:2]} but expected {(reference_height, reference_width)}."
            )
        normalized_frames.append(np.ascontiguousarray(frame))

    # yuv420p requires even spatial resolution. Pad the bottom/right edge if needed.
    if reference_height % 2 != 0 or reference_width % 2 != 0:
        padded_frames = []
        pad_height = reference_height % 2
        pad_width = reference_width % 2
        for frame in normalized_frames:
            padded_frames.append(
                np.pad(
                    frame,
                    ((0, pad_height), (0, pad_width), (0, 0)),
                    mode="edge",
                )
            )
        normalized_frames = padded_frames
        reference_height += pad_height
        reference_width += pad_width

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    writer = imageio.get_writer(
        str(path),
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
        ffmpeg_log_level="error",
        output_params=["-crf", "21"],
    )
    try:
        for frame in normalized_frames:
            writer.append_data(frame)
    except Exception as exc:
        frame_shapes = [frame.shape for frame in normalized_frames[:3]]
        if len(normalized_frames) > 3:
            frame_shapes.append(("...", len(normalized_frames)))
        raise OSError(
            f"{exc}\n[save_video] ffmpeg={ffmpeg_path}, "
            f"fps={fps}, first_frame_shape={(reference_height, reference_width, reference_channels)}, "
            f"sample_frame_shapes={frame_shapes}, output={path}"
        ) from exc
    finally:
        writer.close()
