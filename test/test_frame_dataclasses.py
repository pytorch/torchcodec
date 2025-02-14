import pytest
import torch
from torchcodec import Frame, FrameBatch


def test_frame_unpacking():
    data, pts_seconds, duration_seconds = Frame(torch.rand(3, 4, 5), 2, 3)  # noqa


# TODO: put back
# def test_frame_error():
#     with pytest.raises(ValueError, match="data must be 3-dimensional"):
#         Frame(
#             data=torch.rand(1, 2),
#             pts_seconds=1,
#             duration_seconds=1,
#         )
#     with pytest.raises(ValueError, match="data must be 3-dimensional"):
#         Frame(
#             data=torch.rand(1, 2, 3, 4),
#             pts_seconds=1,
#             duration_seconds=1,
#         )


def test_framebatch_error():
    with pytest.raises(ValueError, match="data must be at least 3-dimensional"):
        FrameBatch(
            data=torch.rand(2, 3),
            pts_seconds=torch.rand(1),
            duration_seconds=torch.rand(1),
        )

    # Note: this is expected to fail because pts_seconds and duration_seconds
    # are expected to have a shape of size([]) instead of size([1]).
    with pytest.raises(
        ValueError, match="leading dimensions of the inputs do not match"
    ):
        FrameBatch(
            data=torch.rand(1, 2, 3),
            pts_seconds=torch.rand(1),
            duration_seconds=torch.rand(1),
        )

    with pytest.raises(
        ValueError, match="leading dimensions of the inputs do not match"
    ):
        FrameBatch(
            data=torch.rand(3, 4, 2, 1),
            pts_seconds=torch.rand(3),  # ok
            duration_seconds=torch.rand(2),  # bad
        )

    with pytest.raises(
        ValueError, match="leading dimensions of the inputs do not match"
    ):
        FrameBatch(
            data=torch.rand(3, 4, 2, 1),
            pts_seconds=torch.rand(2),  # bad
            duration_seconds=torch.rand(3),  # ok
        )

    with pytest.raises(
        ValueError, match="leading dimensions of the inputs do not match"
    ):
        FrameBatch(
            data=torch.rand(5, 3, 4, 2, 1),
            pts_seconds=torch.rand(5, 3),  # ok
            duration_seconds=torch.rand(5, 2),  # bad
        )

    with pytest.raises(
        ValueError, match="leading dimensions of the inputs do not match"
    ):
        FrameBatch(
            data=torch.rand(5, 3, 4, 2, 1),
            pts_seconds=torch.rand(5, 2),  # bad
            duration_seconds=torch.rand(5, 3),  # ok
        )


def test_framebatch_iteration():
    T, N, C, H, W = 7, 6, 3, 2, 4

    fb = FrameBatch(
        data=torch.rand(T, N, C, H, W),
        pts_seconds=torch.rand(T, N),
        duration_seconds=torch.rand(T, N),
    )

    for sub_fb in fb:
        assert isinstance(sub_fb, FrameBatch)
        assert sub_fb.data.shape == (N, C, H, W)
        assert sub_fb.pts_seconds.shape == (N,)
        assert sub_fb.duration_seconds.shape == (N,)
        for frame in sub_fb:
            assert isinstance(frame, FrameBatch)
            assert frame.data.shape == (C, H, W)
            # pts_seconds and duration_seconds are 0-dim tensors but they still
            # contain a value
            assert frame.pts_seconds.shape == tuple()
            assert frame.duration_seconds.shape == tuple()
            frame.pts_seconds.item()
            frame.duration_seconds.item()

    # Check unpacking behavior
    first_sub_fb, *_ = fb
    assert isinstance(first_sub_fb, FrameBatch)


def test_framebatch_indexing():
    T, N, C, H, W = 7, 6, 3, 2, 4

    fb = FrameBatch(
        data=torch.rand(T, N, C, H, W),
        pts_seconds=torch.rand(T, N),
        duration_seconds=torch.rand(T, N),
    )

    for i in range(len(fb)):
        assert isinstance(fb[i], FrameBatch)
        assert fb[i].data.shape == (N, C, H, W)
        assert fb[i].pts_seconds.shape == (N,)
        assert fb[i].duration_seconds.shape == (N,)
        for j in range(len(fb[i])):
            frame = fb[i][j]
            assert isinstance(frame, FrameBatch)
            assert frame.data.shape == (C, H, W)
            # pts_seconds and duration_seconds are 0-dim tensors but they still
            # contain a value
            assert frame.pts_seconds.shape == tuple()
            assert frame.duration_seconds.shape == tuple()
            frame.pts_seconds.item()
            frame.duration_seconds.item()

    fb_fancy = fb[torch.arange(3)]
    assert isinstance(fb_fancy, FrameBatch)
    assert fb_fancy.data.shape == (3, N, C, H, W)

    fb_fancy = fb[[[0], [1]]]  # select T=0 and N=1.
    assert isinstance(fb_fancy, FrameBatch)
    assert fb_fancy.data.shape == (1, C, H, W)
