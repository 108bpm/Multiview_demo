from __future__ import annotations

from typing import Any, Literal, Optional, Tuple, Union, List

import torch
from pydantic import Field, BaseModel

from .base import ModalityTransform


ArmSide = Literal["left", "right"]


class BimanualPadAndImagePadTransform(ModalityTransform):
    """
    Pad heterogeneous robot datasets into a unified bimanual layout + pad multi-view images.

    Canonical arm layout (feature dimension order):
      [LEFT_ARM_FEATURES, RIGHT_ARM_FEATURES]

    - Dual-arm sample: state/action already contains both arms -> keep order as-is (assumed left then right).
    - Single-arm sample: place it into one slot (default RIGHT), and set the missing slot to padding.

    Also pads image views to a fixed max_views and produces a view mask.

    Expected data fields (if present):
      - data["state"]:  torch.Tensor [T, D_state]
      - data["action"]: torch.Tensor [T, D_action]
      - data["images"]: torch.Tensor [V, C, H, W]  (preferred)
        OR data["all_images"]: List[torch.Tensor] each [C, H, W]
    Outputs (if corresponding inputs exist):
      - data["state"], data["state_mask"], data["state_arm_mask"]
      - data["action"], data["action_mask"], data["action_arm_mask"]
      - data["images"], data["image_view_mask"]
    """

    # --- state/action canonicalization ---
    arm_state_dim: int = Field(..., description="Per-arm state feature dim (single-arm D_state).")
    arm_action_dim: int = Field(..., description="Per-arm action feature dim (single-arm D_action).")
    single_arm_placement: ArmSide = Field(
        default="right",
        description="Where to place single-arm features in the bimanual layout.",
    )

    # Final padded dims (can be >= 2 * per-arm dims; extra dims are treated as 'global/other' and padded)
    max_state_dim: int = Field(..., description="Final padded state dim.")
    max_action_dim: int = Field(..., description="Final padded action dim.")

    # --- image padding ---
    max_views: int = Field(..., description="Pad image views to this fixed number of views.")
    image_hw: Tuple[int, int] = Field(default=(224, 224), description="Expected (H, W) for images (already resized upstream).")

    # Padding values (usually 0 is best if you normalize + use masks correctly)
    pad_value_state: float = 0.0
    pad_value_action: float = 0.0
    pad_value_image: float = 0.0

    # Kept for compatibility with ModalityTransform API (not used here)
    apply_to: list[str] = Field(default_factory=list)

    def _pad_last_dim(
        self,
        x: torch.Tensor,  # [T, D]
        max_dim: int,
        pad_value: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad (or truncate) feature dim to max_dim and return (x_padded, dim_mask[ max_dim ])."""
        assert x.ndim == 2, f"Expected [T, D], got {tuple(x.shape)}"
        T, D = x.shape
        dim_mask = torch.zeros(max_dim, dtype=torch.bool, device=x.device)

        keep = min(D, max_dim)
        dim_mask[:keep] = True

        if D == max_dim:
            return x, dim_mask
        if D > max_dim:
            return x[:, :max_dim], dim_mask

        pad = torch.full((T, max_dim - D), pad_value, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=-1), dim_mask

    def _to_bimanual_layout(
        self,
        x: torch.Tensor,              # [T, D]
        per_arm_dim: int,
        max_dim: int,
        pad_value: float,
        placement: ArmSide,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert x to canonical [L, R] layout (length 2*per_arm_dim) then pad to max_dim.

        Returns:
          x_out:      [T, max_dim]
          dim_mask:   [max_dim]     True = real feature, False = padded
          arm_mask:   [2]           True = arm present
        """
        assert x.ndim == 2, f"Expected [T, D], got {tuple(x.shape)}"
        T, D = x.shape
        dev = x.device

        L = per_arm_dim
        bimanual_dim = 2 * L

        arm_mask = torch.zeros(2, dtype=torch.bool, device=dev)  # [left, right]
        # Build canonical bimanual tensor first (size bimanual_dim), then pad to max_dim.
        base = torch.full((T, bimanual_dim), pad_value, dtype=x.dtype, device=dev)

        if D == L:
            # single-arm
            if placement == "left":
                base[:, :L] = x
                arm_mask[0] = True
            else:
                base[:, L:2 * L] = x
                arm_mask[1] = True
            bimanual = base
            bimanual_mask = torch.zeros(bimanual_dim, dtype=torch.bool, device=dev)
            if placement == "left":
                bimanual_mask[:L] = True
            else:
                bimanual_mask[L:2 * L] = True

        elif D == bimanual_dim:
            # dual-arm
            bimanual = x  # assume already [L, R] in order
            arm_mask[:] = True
            bimanual_mask = torch.ones(bimanual_dim, dtype=torch.bool, device=dev)

        else:
            # Unknown schema (e.g., includes global/base features). Fallback:
            # pad/truncate directly to max_dim; arm_mask is best-effort inference.
            x_out, dim_mask = self._pad_last_dim(x, max_dim=max_dim, pad_value=pad_value)
            if D == L:
                arm_mask[1 if placement == "right" else 0] = True
            elif D >= bimanual_dim:
                arm_mask[:] = True
            return x_out, dim_mask, arm_mask

        # Now pad bimanual to max_dim (may have extra global dims reserved)
        if max_dim < bimanual_dim:
            # truncate (not ideal; configure max_dim >= 2*per_arm_dim)
            x_out = bimanual[:, :max_dim]
            dim_mask = bimanual_mask[:max_dim].clone()
            return x_out, dim_mask, arm_mask

        if max_dim == bimanual_dim:
            return bimanual, bimanual_mask, arm_mask

        pad = torch.full((T, max_dim - bimanual_dim), pad_value, dtype=x.dtype, device=dev)
        x_out = torch.cat([bimanual, pad], dim=-1)
        dim_mask = torch.cat(
            [bimanual_mask, torch.zeros(max_dim - bimanual_dim, dtype=torch.bool, device=dev)],
            dim=0,
        )
        return x_out, dim_mask, arm_mask

    def _ensure_images_tensor(self, data: dict[str, Any]) -> Optional[torch.Tensor]:
        """
        Return images as [V, C, H, W] if present, else None.
        Accepts:
          - data["images"]: torch.Tensor [V, C, H, W]
          - data["all_images"]: List[torch.Tensor] each [C, H, W]
        """
        if "images" in data and data["images"] is not None:
            img = data["images"]
            assert isinstance(img, torch.Tensor), f'Expected data["images"] torch.Tensor, got {type(img)}'
            assert img.ndim == 4, f'Expected [V,C,H,W], got {tuple(img.shape)}'
            return img

        if "all_images" in data and data["all_images"] is not None:
            imgs = data["all_images"]
            assert isinstance(imgs, list), f'Expected data["all_images"] list, got {type(imgs)}'
            assert len(imgs) > 0, "all_images is empty"
            for i, t in enumerate(imgs):
                assert isinstance(t, torch.Tensor), f"all_images[{i}] is not torch.Tensor: {type(t)}"
                assert t.ndim == 3, f"all_images[{i}] expected [C,H,W], got {tuple(t.shape)}"
            return torch.stack(imgs, dim=0)

        return None

    def _pad_images_views(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad images along view dimension to max_views.
        images: [V, C, H, W]
        Returns:
          images_out: [max_views, C, H, W]
          view_mask:  [max_views] True for real views
        """
        assert images.ndim == 4, f"Expected [V,C,H,W], got {tuple(images.shape)}"
        V, C, H, W = images.shape
        expH, expW = self.image_hw
        # Not resizing here: assume upstream does resize; we just sanity-check if you want.
        if (H, W) != (expH, expW):
            # Don’t hard fail; but it’s usually better to keep consistent.
            # If you prefer strict, change to assert.
            pass

        dev = images.device
        view_mask = torch.zeros(self.max_views, dtype=torch.bool, device=dev)
        keep = min(V, self.max_views)
        view_mask[:keep] = True

        if V == self.max_views:
            return images, view_mask
        if V > self.max_views:
            return images[: self.max_views], view_mask

        pad = torch.full(
            (self.max_views - V, C, H, W),
            self.pad_value_image,
            dtype=images.dtype,
            device=dev,
        )
        return torch.cat([images, pad], dim=0), view_mask

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        # ---- state ----
        if "state" in data and data["state"] is not None:
            state = data["state"]
            assert isinstance(state, torch.Tensor), f"Expected torch.Tensor for state, got {type(state)}"
            state_out, state_mask, state_arm_mask = self._to_bimanual_layout(
                x=state,
                per_arm_dim=self.arm_state_dim,
                max_dim=self.max_state_dim,
                pad_value=self.pad_value_state,
                placement=self.single_arm_placement,
            )
            data["state"] = state_out
            data["state_mask"] = state_mask          # [max_state_dim]
            data["state_arm_mask"] = state_arm_mask  # [2] (left,right)

        # ---- action ----
        if "action" in data and data["action"] is not None:
            action = data["action"]
            assert isinstance(action, torch.Tensor), f"Expected torch.Tensor for action, got {type(action)}"
            action_out, action_mask, action_arm_mask = self._to_bimanual_layout(
                x=action,
                per_arm_dim=self.arm_action_dim,
                max_dim=self.max_action_dim,
                pad_value=self.pad_value_action,
                placement=self.single_arm_placement,
            )
            data["action"] = action_out
            data["action_mask"] = action_mask
            data["action_arm_mask"] = action_arm_mask

        # ---- images ----
        images = self._ensure_images_tensor(data)
        if images is not None:
            images_out, view_mask = self._pad_images_views(images)
            data["images"] = images_out
            data["image_view_mask"] = view_mask  # [max_views]
            # Optional: if you still keep list form upstream, you can delete it here
            # data.pop("all_images", None)

        return data


class BimanualPadTransform(ModalityTransform):
    """
    Pad heterogeneous robot datasets into a unified bimanual layout WITH masks,
    and WITHOUT any image processing.

    Canonical arm layout (feature dim order):
      [LEFT_ARM_FEATURES, RIGHT_ARM_FEATURES]

    Behavior:
      - If D == per_arm_dim (single-arm):
          place into `single_arm_placement` slot and pad the other slot with pad_value,
          mask marks only the present arm dims as True.
      - If D == 2*per_arm_dim (dual-arm):
          keep as-is (assumed already [left, right]),
          mask marks both arm dims as True.
      - Otherwise:
          fallback to simple pad/truncate to max_dim,
          mask marks the first min(D, max_dim) dims as True.

    Expected fields:
      - data["state"]:  torch.Tensor [T, D_state] (optional)
      - data["action"]: torch.Tensor [T, D_action] (optional)

    Outputs (if input present):
      - data["state"]:       [T, max_state_dim]
      - data["state_mask"]:  [max_state_dim]  (True = real dim, False = padded)
      - data["action"]:      [T, max_action_dim]
      - data["action_mask"]: [max_action_dim]
    """

    # --- per-arm dims ---
    arm_state_dim: int = Field(..., description="Per-arm state feature dim (single-arm).")
    arm_action_dim: int = Field(..., description="Per-arm action feature dim (single-arm).")

    # --- final dims ---
    max_state_dim: int = Field(..., description="Final padded state dim.")
    max_action_dim: int = Field(..., description="Final padded action dim.")

    single_arm_placement: ArmSide = Field(default="right")

    pad_value_state: float = 0.0
    pad_value_action: float = 0.0

    # kept for compatibility
    apply_to: list[str] = Field(default_factory=list)

    def _pad_last_dim_with_mask(
        self, x: torch.Tensor, max_dim: int, pad_value: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad/truncate last dim to max_dim.
        Returns:
          x_out:  [T, max_dim]
          mask:   [max_dim] True for real dims
        """
        assert x.ndim == 2, f"Expected [T, D], got {tuple(x.shape)}"
        T, D = x.shape
        keep = min(D, max_dim)
        mask = torch.zeros(max_dim, dtype=torch.bool, device=x.device)
        mask[:keep] = True

        if D == max_dim:
            return x, mask
        if D > max_dim:
            return x[:, :max_dim], mask

        pad = torch.full((T, max_dim - D), pad_value, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=-1), mask

    def _to_bimanual_with_mask(
        self, x: torch.Tensor, per_arm_dim: int, max_dim: int, pad_value: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert to canonical [L, R] (2*per_arm_dim) when possible, then pad/truncate to max_dim.
        Returns:
          x_out: [T, max_dim]
          mask:  [max_dim]
        """
        assert x.ndim == 2, f"Expected [T, D], got {tuple(x.shape)}"
        T, D = x.shape
        dev = x.device

        bimanual_dim = 2 * per_arm_dim

        if D == per_arm_dim:
            # single-arm -> build [L, R]
            base = torch.full((T, bimanual_dim), pad_value, dtype=x.dtype, device=dev)
            bmask = torch.zeros(bimanual_dim, dtype=torch.bool, device=dev)

            if self.single_arm_placement == "left":
                base[:, :per_arm_dim] = x
                bmask[:per_arm_dim] = True
            else:
                base[:, per_arm_dim:] = x
                bmask[per_arm_dim:] = True

            x_bi = base

            # pad to max_dim
            if max_dim == bimanual_dim:
                return x_bi, bmask
            if max_dim < bimanual_dim:
                return x_bi[:, :max_dim], bmask[:max_dim]
            pad = torch.full((T, max_dim - bimanual_dim), pad_value, dtype=x.dtype, device=dev)
            x_out = torch.cat([x_bi, pad], dim=-1)
            mask = torch.cat([bmask, torch.zeros(max_dim - bimanual_dim, dtype=torch.bool, device=dev)], dim=0)
            return x_out, mask

        if D == bimanual_dim:
            # dual-arm -> keep, full mask on bimanual dims
            bmask = torch.ones(bimanual_dim, dtype=torch.bool, device=dev)

            if max_dim == bimanual_dim:
                return x, bmask
            if max_dim < bimanual_dim:
                return x[:, :max_dim], bmask[:max_dim]
            pad = torch.full((T, max_dim - bimanual_dim), pad_value, dtype=x.dtype, device=dev)
            x_out = torch.cat([x, pad], dim=-1)
            mask = torch.cat([bmask, torch.zeros(max_dim - bimanual_dim, dtype=torch.bool, device=dev)], dim=0)
            return x_out, mask

        # unknown layout -> do not force bimanual; best-effort pad/truncate + mask first dims
        return self._pad_last_dim_with_mask(x, max_dim=max_dim, pad_value=pad_value)

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        # ---- state ----
        if "state" in data and data["state"] is not None:
            state = data["state"]
            assert isinstance(state, torch.Tensor), f"Expected torch.Tensor for state, got {type(state)}"
            state_out, state_mask = self._to_bimanual_with_mask(
                x=state,
                per_arm_dim=self.arm_state_dim,
                max_dim=self.max_state_dim,
                pad_value=self.pad_value_state,
            )
            data["state"] = state_out
            data["state_mask"] = state_mask  # [max_state_dim] even for dual-arm

        # ---- action ----
        if "action" in data and data["action"] is not None:
            action = data["action"]
            assert isinstance(action, torch.Tensor), f"Expected torch.Tensor for action, got {type(action)}"
            action_out, action_mask = self._to_bimanual_with_mask(
                x=action,
                per_arm_dim=self.arm_action_dim,
                max_dim=self.max_action_dim,
                pad_value=self.pad_value_action,
            )
            data["action"] = action_out
            data["action_mask"] = action_mask  # [max_action_dim] even for dual-arm

        return data