// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "DecodeGif.h"

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/util/Exception.h>

#include "StableABICompat.h"

#if !TORCHCODEC_ENABLE_GIF

namespace facebook::torchcodec {

torch::stable::Tensor decode_gif(
    [[maybe_unused]] const torch::stable::Tensor& input,
    [[maybe_unused]] int64_t mode) {
  STD_TORCH_CHECK(
      false,
      "decode_gif: torchcodec was not compiled with GIF support. Rebuild "
      "torchcodec with TORCHCODEC_BUILD_GIF=1. If you see this error in a "
      "prebuilt wheel, please report it to the TorchCodec repo.");
}

} // namespace facebook::torchcodec

#else

#include <torch/headeronly/core/MemoryFormat.h>

#include <algorithm>
#include <cstdint>
#include <optional>

#include "ImageCommon.h"
#include "giflib/gif_lib.h"

namespace facebook::torchcodec {

namespace {

struct SourceCtx {
  const uint8_t* ptr; // current read position in the input tensor
  size_t count; // number of bytes left to read
};

// Reader passed to DGifOpen(): giflib calls it to pull `bytes` encoded bytes
// into `output`. We serve them from the input tensor (see SourceCtx), advancing
// as we go, and return the number of bytes actually read.
int read_callback(GifFileType* gif_file, GifByteType* output, int bytes) {
  // UserData was set in DGifOpen().
  auto* source_ctx = static_cast<SourceCtx*>(gif_file->UserData);
  size_t num_bytes_to_read =
      std::min(static_cast<size_t>(bytes), source_ctx->count);
  std::copy(source_ctx->ptr, source_ctx->ptr + num_bytes_to_read, output);
  source_ctx->ptr += num_bytes_to_read;
  source_ctx->count -= num_bytes_to_read;
  return static_cast<int>(num_bytes_to_read);
}

} // namespace

torch::stable::Tensor decode_gif(
    const torch::stable::Tensor& input,
    int64_t mode) {
  // GIF decoding model (see https://giflib.sourceforge.net/whatsinagif/):
  //
  // A GIF is a shared canvas (the "logical screen") onto which one or more
  // FRAMES are painted in order. Each frame is a palette image (<= 256 colors)
  // that may be smaller than the canvas and placed at an offset, and it is
  // composited onto the running canvas -- so frame i as displayed is the
  // accumulation of all frames so far, not frame i's pixels alone.
  //
  // Two per-frame settings (both in the frame's Graphic Control Extension)
  // drive that compositing:
  // - TRANSPARENCY: one palette index can be marked transparent; those pixels
  //   are "not drawn", so whatever is already on the canvas shows through. It
  //   is binary (fully transparent or fully opaque).
  // - DISPOSAL METHOD: what to do with the canvas AFTER this frame is shown, to
  //   prepare for the next one:
  //     - DISPOSE_DO_NOT leaves it (the next frame draws on top)
  //     - DISPOSE_BACKGROUND clears this frame's area to the background color
  //       (a palette index from the logical screen descriptor)
  //     - DISPOSE_PREVIOUS restores the canvas to its state before this frame.
  //
  // So frame i's starting canvas is decided by frame (i-1)'s disposal method,
  // and transparency then lets frame i reveal it. Below we rebuild that canvas
  // per frame and write each composited frame into the output tensor.
  //
  // LibGif API docs: https://giflib.sourceforge.net/intro.html

  auto contig_input = validate_encoded_data(input);

  int error = D_GIF_SUCCEEDED;

  // We're using DGifOpen. The other entrypoints of libgif are DGifOpenFileName
  // and DGifOpenFileHandle but we don't want to use those, since we need to
  // read the encoded bytes from a tensor of encoded bytes, not from a file (for
  // consistency with existing jpeg and png decoders). Using DGifOpen is the
  // only way to read from a custom source, via the read_callback reader above.

  // TODO: We are potentially doing an unnecessary copy of the encoded bytes:
  // - 1 copy from file to tensor (in the Python _source_to_tensor())
  // - 1 copy from tensor to GIFLIB buffers (in read_callback())
  // Since we're vendoring GIFLIB we can potentially modify the calls to
  // InternalRead() and just set the `buf` pointer to the tensor data directly.
  // That might even save allocation of those buffers.
  // If we do that, we'd have to make sure the buffers are never written to by
  // GIFLIB, otherwise we'd be overriding the tensor data.
  SourceCtx source_ctx{
      .ptr = contig_input.const_data_ptr<uint8_t>(),
      .count = static_cast<size_t>(contig_input.numel())};
  GifFileType* gif_file =
      DGifOpen(static_cast<void*>(&source_ctx), read_callback, &error);

  STD_TORCH_CHECK(
      (gif_file != nullptr) && (error == D_GIF_SUCCEEDED),
      "DGifOpen() failed - ",
      error);

  if (DGifSlurp(gif_file) == GIF_ERROR) {
    auto slurp_error = gif_file->Error;
    DGifCloseFile(gif_file, &error);
    STD_TORCH_CHECK(false, "DGifSlurp() failed - ", slurp_error);
  }
  auto num_images = gif_file->ImageCount;

  // This check should already be done within DGifSlurp(), just to be safe.
  STD_TORCH_CHECK(
      num_images > 0, "GIF file should contain at least one image!");

  GifColorType bg{0, 0, 0};
  if (gif_file->SColorMap &&
      // out of range background falls back to black
      gif_file->SBackGroundColor < gif_file->SColorMap->ColorCount) {
    bg = gif_file->SColorMap->Colors[gif_file->SBackGroundColor];
  }

  bool has_transparency = false;
  for (int i = 0; i < num_images; ++i) {
    GraphicsControlBlock gcb;
    if (DGifSavedExtensionToGCB(gif_file, i, &gcb) == GIF_OK &&
        gcb.TransparentColor != NO_TRANSPARENT_COLOR) {
      has_transparency = true;
      break;
    }
  }

  // We represent transparency differently per output mode, and this is where we
  // diverge from Pillow (see the primer above for how transparency/disposal
  // build the canvas that transparent pixels reveal):
  // - RGB_ALPHA, and UNCHANGED when the GIF has transparency: background/
  //   uncovered/disposed regions become alpha 0 (i.e. transparent); since GIF
  //   transparency is binary, the alpha we emit is always 0 or 255. This
  //   matches Pillow's RGBA output exactly (alpha, and RGB wherever opaque).
  // - RGB: there is no alpha, so those regions instead get the GIF background
  //   color via fill_background(). This is spec-faithful (the background color
  //   is what uncovered pixels should show) but differs from Pillow's
  //   convert("RGB"): Pillow ignores the background color and, for a
  //   transparent pixel with nothing opaque behind it, shows the transparent
  //   index's own palette color. We deliberately do not match Pillow here. (A
  //   transparent pixel over an opaque previous frame shows that frame in both,
  //   so we only diverge on background/uncovered regions.)
  bool emit_alpha =
      !should_return_rgb(static_cast<ImageReadMode>(mode), has_transparency);
  int num_output_channels = emit_alpha ? 4 : 3;

  // The GIFLIB docs say that the canvas's height and width are potentially
  // ignored by modern viewers, so to be on the safe side we set the output
  // height to max(canvas_height, first_image_height). Same for width.
  // https://giflib.sourceforge.net/whatsinagif/bits_and_bytes.html
  auto output_h =
      std::max(gif_file->SHeight, gif_file->SavedImages[0].ImageDesc.Height);
  auto output_w =
      std::max(gif_file->SWidth, gif_file->SavedImages[0].ImageDesc.Width);

  auto output = torch::stable::empty(
      {static_cast<int64_t>(num_images),
       num_output_channels,
       static_cast<int64_t>(output_h),
       static_cast<int64_t>(output_w)},
      kStableUInt8,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      torch::headeronly::MemoryFormat::ChannelsLast);
  auto output_a = mutable_accessor<uint8_t, 4>(output);

  // Clears frame i over the rectangle [top, top + height) x [left, left +
  // width) (clipped to the canvas). For RGB output this paints the GIF
  // background color; for RGBA output it clears to fully transparent (matching
  // web browsers / Pillow, which ignore the background color for transparent
  // GIFs).
  auto fill_background = [&](int i, int top, int left, int height, int width) {
    for (int h = 0; h < height; ++h) {
      const auto y = static_cast<int64_t>(top) + h;
      if (y < 0 || y >= output_h) {
        continue;
      }
      for (int w = 0; w < width; ++w) {
        const auto x = static_cast<int64_t>(left) + w;
        if (x < 0 || x >= output_w) {
          continue;
        }
        if (emit_alpha) {
          output_a[i][0][y][x] = 0;
          output_a[i][1][y][x] = 0;
          output_a[i][2][y][x] = 0;
          output_a[i][3][y][x] = 0;
        } else {
          output_a[i][0][y][x] = bg.Red;
          output_a[i][1][y][x] = bg.Green;
          output_a[i][2][y][x] = bg.Blue;
        }
      }
    }
  };

  // Loop-carried canvas state (used by prepare_canvas): the previous frame's
  // disposal method and image rectangle, plus a snapshot of the canvas taken
  // before a DISPOSE_PREVIOUS frame is drawn (lazily allocated, shape
  // (1, C, H, W)) so the following frame can restore it.
  int prev_disposal = DISPOSAL_UNSPECIFIED;
  GifImageDesc prev_desc{};
  torch::stable::Tensor restore_point;
  bool have_restore_point = false;

  // Stage 1: set output[i] to the base canvas frame i is drawn onto, per the
  // *previous* frame's disposal method (see the primer). DISPOSE_BACKGROUND
  // clears only the previous frame's rectangle (keeping the rest of the
  // accumulated canvas); DISPOSE_PREVIOUS restores the snapshot taken before
  // that frame; the first frame starts from a fully cleared canvas. Also
  // snapshots this frame's base if it will itself be disposed with
  // DISPOSE_PREVIOUS.
  auto prepare_canvas = [&](int i, const GraphicsControlBlock& gcb) {
    if (i == 0) {
      // Clear the whole first-frame canvas, not just the logical screen: when
      // the first image is larger than the logical screen the output is sized
      // to the image (output_h/output_w above), and any pixel the image leaves
      // transparent there would otherwise be left uninitialized.
      fill_background(0, 0, 0, output_h, output_w);
    } else if (prev_disposal == DISPOSE_BACKGROUND) {
      copy_frame(output, i, output, i - 1);
      fill_background(
          i, prev_desc.Top, prev_desc.Left, prev_desc.Height, prev_desc.Width);
    } else if (prev_disposal == DISPOSE_PREVIOUS && have_restore_point) {
      copy_frame(output, i, restore_point, 0);
    } else {
      // DISPOSE_DO_NOT or DISPOSAL_UNSPECIFIED: just copy the previous canvas.
      copy_frame(output, i, output, i - 1);
    }

    // Save current canvas state for next frame to restore.
    if (gcb.DisposalMode == DISPOSE_PREVIOUS) {
      if (!have_restore_point) {
        restore_point = torch::stable::empty(
            {1,
             num_output_channels,
             static_cast<int64_t>(output_h),
             static_cast<int64_t>(output_w)},
            kStableUInt8);
        have_restore_point = true;
      }
      copy_frame(restore_point, 0, output, i);
    }
  };

  // Stage 2: draw frame i's opaque pixels on top of its prepared base canvas.
  // Transparent pixels are skipped so the base shows through, and pixels that
  // fall outside the output tensor (a frame larger than the first, or with a
  // non-zero offset) are dropped.
  auto draw_frame_over_canvas = [&](int i,
                                    const SavedImage& img,
                                    const GraphicsControlBlock& gcb) {
    const GifImageDesc& desc = img.ImageDesc;
    const ColorMapObject* cmap =
        desc.ColorMap ? desc.ColorMap : gif_file->SColorMap;
    STD_TORCH_CHECK(
        cmap != nullptr,
        "Global and local color maps are missing. This should never happen!");
    for (int h = 0; h < desc.Height; ++h) {
      const auto y = static_cast<int64_t>(desc.Top) + h;
      if (y < 0 || y >= output_h) {
        continue;
      }
      for (int w = 0; w < desc.Width; ++w) {
        const auto x = static_cast<int64_t>(desc.Left) + w;
        if (x < 0 || x >= output_w) {
          continue;
        }
        auto c = img.RasterBits[h * desc.Width + w];
        if (c == gcb.TransparentColor) {
          continue;
        }
        GifColorType rgb =
            c < cmap->ColorCount ? cmap->Colors[c] : GifColorType{0, 0, 0};
        output_a[i][0][y][x] = rgb.Red;
        output_a[i][1][y][x] = rgb.Green;
        output_a[i][2][y][x] = rgb.Blue;
        if (emit_alpha) {
          output_a[i][3][y][x] = 255;
        }
      }
    }
  };

  for (int i = 0; i < num_images; ++i) {
    const SavedImage& img = gif_file->SavedImages[i];
    GraphicsControlBlock gcb;
    DGifSavedExtensionToGCB(gif_file, i, &gcb);

    prepare_canvas(i, gcb);
    draw_frame_over_canvas(i, img, gcb);

    prev_disposal = gcb.DisposalMode;
    prev_desc = img.ImageDesc;
  }

  // Remove the batch dim if there's only one image, so a still GIF decodes to a
  // (C, H, W) tensor like the other image decoders.
  if (num_images == 1) {
    output = select_row(output, 0);
  }

  DGifCloseFile(gif_file, &error);
  STD_TORCH_CHECK(error == D_GIF_SUCCEEDED, "DGifCloseFile() failed - ", error);

  return output;
}

} // namespace facebook::torchcodec

#endif // !TORCHCODEC_ENABLE_GIF
