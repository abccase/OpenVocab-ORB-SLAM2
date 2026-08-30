from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_renderer_contract_embeds_masks_and_self_contained_panels() -> None:
    source = (ROOT / "tools/render_visual_acceptance.py").read_text(encoding="utf-8")
    assert "read_cache_frame" in source
    assert "decode_binary_mask_rle" in source
    assert "semantic instance masks" in source
    assert "data:image/" in source
    assert 'src="visual_assets/' in source  # validator must reject local assets
    assert "minimum_render_width_px" in source
