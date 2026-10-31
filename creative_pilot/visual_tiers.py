"""Tiers assigned by an image-capable model (Claude, this development session) reading blind contact sheets.

Tier 1 strong, 2 acceptable, 3 weak, 4 poor/unusable. Judged on legibility, hierarchy, use of canvas space, balance and
grouping/overlap, from the rendered PNG sheets in data/creative_pilot_v1/visual_review/sheets (numbers map to
candidates through key.json, which the assessor did not see in the sheet). These are model judgments, not human labels,
and cover a fixed-stride sample of tasks chosen without looking at candidate quality.
"""

ASSESSOR = "claude_visual_contact_sheet_v1"
TIERS = {
    "000_hero_image_caption_1600x400": {1: 4, 2: 2, 3: 3, 4: 3, 5: 3, 6: 1, 7: 3, 8: 2, 9: 1, 10: 4, 11: 1},
    "004_hero_image_caption_728x90": {1: 1, 2: 4, 3: 4, 4: 3, 5: 3, 6: 3, 7: 1, 8: 2, 9: 3, 10: 1, 11: 2},
    "008_giant_type_stack_90x728": {1: 4, 2: 3, 3: 2, 4: 4, 5: 4, 6: 4, 7: 2, 8: 2, 9: 2, 10: 3, 11: 3},
    "013_logo_tagline_minimal_1600x400": {1: 2, 2: 4, 3: 1, 4: 4, 5: 4, 6: 2, 7: 2, 8: 2, 9: 2, 10: 4, 11: 2},
    "017_offset_product_spotlight_728x90": {1: 4, 2: 4, 3: 4, 4: 1, 5: 2, 6: 3, 7: 1, 8: 2, 9: 3, 10: 2, 11: 3},
    "021_asymmetric_editorial_90x728": {1: 4, 2: 2, 3: 4, 4: 1, 5: 2, 6: 3, 7: 3, 8: 4, 9: 3, 10: 1, 11: 4},
    "026_layered_promo_circles_1600x400": {1: 3, 2: 3, 3: 2, 4: 3, 5: 3, 6: 4, 7: 2, 8: 3, 9: 2, 10: 2, 11: 4},
    "030_diagonal_staggered_blocks_728x90": {1: 1, 2: 2, 3: 1, 4: 4, 5: 4, 6: 4, 7: 2, 8: 4, 9: 2, 10: 4, 11: 3},
    "034_stat_dashboard_90x728": {1: 4, 2: 1, 3: 4, 4: 2, 5: 4, 6: 3, 7: 4, 8: 1, 9: 1, 10: 4, 11: 1},
    "039_feature_trio_1600x400": {1: 2, 2: 4, 3: 2, 4: 2, 5: 1, 6: 4, 7: 1, 8: 3, 9: 1, 10: 3, 11: 4},
    "043_timeline_sequence_728x90": {1: 4, 2: 2, 3: 4, 4: 3, 5: 4, 6: 2, 7: 4, 8: 3, 9: 2, 10: 3, 11: 2},
    "047_poster_split_diagonal_90x728": {1: 2, 2: 2, 3: 2, 4: 4, 5: 4, 6: 3, 7: 4, 8: 3, 9: 2, 10: 3, 11: 3},
    "052_quote_callout_1600x400": {1: 4, 2: 1, 3: 4, 4: 4, 5: 2, 6: 4, 7: 2, 8: 2, 9: 4, 10: 2, 11: 4},
    "056_price_tag_product_728x90": {1: 1, 2: 4, 3: 2, 4: 4, 5: 1, 6: 3, 7: 2, 8: 3, 9: 3, 10: 2, 11: 3},
    "060_magazine_cover_90x728": {1: 4, 2: 4, 3: 2, 4: 1, 5: 3, 6: 2, 7: 4, 8: 3, 9: 2, 10: 2, 11: 4},
    "065_image_tile_gallery_1600x400": {1: 2, 2: 2, 3: 3, 4: 2, 5: 4, 6: 1, 7: 2, 8: 4, 9: 1, 10: 4, 11: 4},
    "069_sidebar_notice_728x90": {1: 2, 2: 4, 3: 4, 4: 3, 5: 1, 6: 3, 7: 3, 8: 2, 9: 3, 10: 3, 11: 3},
    "073_dense_magazine_grid_90x728": {1: 2, 2: 3, 3: 3, 4: 4, 5: 3, 6: 4, 7: 3, 8: 2, 9: 4, 10: 4, 11: 2},
    "078_multi_panel_info_1600x400": {1: 1, 2: 4, 3: 4, 4: 1, 5: 2, 6: 4, 7: 1, 8: 4, 9: 3, 10: 2, 11: 2},
    "082_competing_focal_points_728x90": {1: 4, 2: 4, 3: 4, 4: 3, 5: 2, 6: 2, 7: 1},
    "086_event_lineup_poster_90x728": {1: 1, 2: 4, 3: 4, 4: 1, 5: 2, 6: 4, 7: 1, 8: 4, 9: 3, 10: 2, 11: 4},
    "091_product_catalog_cards_1600x400": {1: 1, 2: 4, 3: 3, 4: 2, 5: 2, 6: 4, 7: 4, 8: 3, 9: 2, 10: 3, 11: 3},
    "095_layered_geometric_poster_728x90": {1: 1, 2: 2, 3: 3, 4: 4, 5: 2, 6: 4, 7: 1, 8: 2, 9: 2, 10: 3, 11: 4},
    "099_layered_geometric_poster_90x728": {1: 1, 2: 2, 3: 4, 4: 4, 5: 3, 6: 2, 7: 2, 8: 2, 9: 3, 10: 4, 11: 4},
}
