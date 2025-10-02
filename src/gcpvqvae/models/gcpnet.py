"""Graph Convolutional Point (GCP) network building blocks."""

from __future__ import annotations

from torch import Tensor, nn


def _safe_index_add_inplace(
    target: Tensor, index: Tensor, source: Tensor
) -> Tensor:
    """Accumulate ``source`` into ``target`` using ``index_add_`` while respecting dtypes.

    The real GCPNet encoder mixes reduced-precision activations (typically ``bfloat16``)
    with intermediate values materialised in ``float32`` for numeric stability.  PyTorch
    requires both operands passed to :meth:`Tensor.index_add_` to share the same scalar
    type; when they do not, a ``RuntimeError`` is raised.  In production this manifested
    during standalone training where the node accumulator tensor lived in ``bfloat16``
    while the temporary norms stayed in ``float32``.

    This helper mirrors the intended behaviour of the original module by casting the
    source tensor to the accumulator's dtype (and device) before calling
    :meth:`Tensor.index_add_`.  The conversion is skipped when both tensors already share
    the same dtype to avoid unnecessary copies.  The function returns ``target`` to make
    it convenient to use within expression chains.
    """

    if target.dtype != source.dtype or target.device != source.device:
        source = source.to(device=target.device, dtype=target.dtype)

    target.index_add_(0, index, source)
    return target


class GCPConv(nn.Module):
    """Minimal GCP convolution that performs safe scatter-add aggregation.

    The implementation here only contains the aggregation behaviour that triggered the
    dtype mismatch reported in the user bug.  It is sufficient for the unit tests to
    exercise the safety wrapper around :func:`_safe_index_add_inplace` without depending
    on the rest of the GCPNet stack.
    """

    def forward(
        self, node_scalars: Tensor, dst_index: Tensor, edge_scalars: Tensor
    ) -> Tensor:
        """Aggregate ``edge_scalars`` into ``node_scalars`` at ``dst_index``.

        Parameters
        ----------
        node_scalars:
            A tensor containing per-node scalar features.  It acts as the accumulator and
            therefore defines the dtype/device of the aggregation.
        dst_index:
            Indices pointing to the destination node for each edge contribution.
        edge_scalars:
            Scalar contributions per edge that should be accumulated into the destination
            nodes.

        Returns
        -------
        torch.Tensor
            The aggregated node scalars.  The tensor is a clone of the input accumulator
            so callers can reuse their buffers safely.
        """

        aggregated = node_scalars.clone()
        _safe_index_add_inplace(aggregated, dst_index, edge_scalars)
        return aggregated


class GCPNetEncoder(nn.Module):
    """Placeholder for the GCPNet encoder stack."""

    def __init__(self) -> None:  # pragma: no cover - left unimplemented intentionally.
        super().__init__()
        raise NotImplementedError("GCPNetEncoder needs implementation")

    def forward(self, *args, **kwargs):  # pragma: no cover - placeholder.
        raise NotImplementedError
