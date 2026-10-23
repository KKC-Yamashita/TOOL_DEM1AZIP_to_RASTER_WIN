# -*- coding: utf-8 -*-
"""
main.py
全体制御
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

import config
from dem_utils import (
    build_dst_transform,
    build_gtiff_profile,
    build_src_transform,
    build_tile_extent_record,
    build_tile_grid_spec_from_root,
    snap_bbox_to_grid,
    collect_intersecting_tiles,
    log_info,
    parse_envelope,
    parse_grid_size,
    parse_sequence_rule_order,
    parse_start_point,
    parse_tuplelist_to_array,
    read_xml_root_from_zip,
    save_tile_extent_parquet,
)


def main() -> None:
    # -----------------------------------------------------
    # 出力先準備
    # -----------------------------------------------------
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # ZIP 一覧取得
    # -----------------------------------------------------
    zip_files = sorted(config.TARGET_ZIP_DIR.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"ZIPファイルが見つかりません: {config.TARGET_ZIP_DIR}")

    log_info("ZIP検索開始...")

    # -----------------------------------------------------
    # 対象範囲と交差する XML を収集
    # -----------------------------------------------------
    target_tiles = collect_intersecting_tiles(zip_files, config.TARGET_BBOX)
    log_info(f"交差XML数: {len(target_tiles)}")

    if not target_tiles:
        raise RuntimeError("指定範囲に交差するXMLが見つかりませんでした。")

    # -----------------------------------------------------
    # 1枚目から格子仕様を取得
    # -----------------------------------------------------
    first_root = read_xml_root_from_zip(
        target_tiles[0].zip_path,
        target_tiles[0].xml_name,
    )
    base_grid = build_tile_grid_spec_from_root(first_root)
    anchor_west, _, _, anchor_north = parse_envelope(first_root)

    log_info(
        f"tile grid: nx={base_grid.nx}, ny={base_grid.ny}, "
        f"dx={base_grid.dx:.15f}, dy={base_grid.dy:.15f}"
    )

    # -----------------------------------------------------
    # 出力グリッド作成（スナップ処理）
    # -----------------------------------------------------
    (
        snapped_west,
        snapped_south,
        snapped_east,
        snapped_north,
        out_width,
        out_height,
    ) = snap_bbox_to_grid(
        west=config.WEST,
        south=config.SOUTH,
        east=config.EAST,
        north=config.NORTH,
        anchor_west=anchor_west,
        anchor_north=anchor_north,
        dx=base_grid.dx,
        dy=base_grid.dy,
    )

    log_info(f"Requested bounds : W={config.WEST}, S={config.SOUTH}, E={config.EAST}, N={config.NORTH}")
    log_info(f"Snapped bounds   : W={snapped_west}, S={snapped_south}, E={snapped_east}, N={snapped_north}")
    log_info(f"Output raster dim: width={out_width}, height={out_height}")

    out_array = np.full(
        (out_height, out_width),
        config.NODATA_VALUE,
        dtype=np.float32,
    )

    dst_transform = build_dst_transform(
        west=snapped_west,
        north=snapped_north,
        dx=base_grid.dx,
        dy=base_grid.dy,
    )

    tile_extent_records: list[dict] = []

    # -----------------------------------------------------
    # 各タイルを読み、出力配列へモザイク
    # -----------------------------------------------------
    for idx, tile in enumerate(target_tiles, start=1):
        log_info(f"[{idx}/{len(target_tiles)}] {tile.zip_path.name} :: {tile.xml_name}")

        root = read_xml_root_from_zip(tile.zip_path, tile.xml_name)

        west, south, east, north = parse_envelope(root)
        nx, ny = parse_grid_size(root)
        start_x, start_y = parse_start_point(root)
        order = parse_sequence_rule_order(root)

        if nx != base_grid.nx or ny != base_grid.ny:
            raise RuntimeError(
                f"格子点数が一致しません: {tile.xml_name} "
                f"(nx, ny)=({nx}, {ny})"
            )

        src_array = parse_tuplelist_to_array(
            root=root,
            nx=nx,
            ny=ny,
            nodata_value=config.NODATA_VALUE,
        )

        src_transform = build_src_transform(
            west=west,
            south=south,
            east=east,
            north=north,
            nx=nx,
            ny=ny,
        )

        # 想定タイル矩形レコード
        tile_extent_records.append(
            build_tile_extent_record(
                zip_name=tile.zip_path.name,
                xml_name=tile.xml_name,
                west=west,
                south=south,
                east=east,
                north=north,
                start_x=start_x,
                start_y=start_y,
                order=order,
                nx=nx,
                ny=ny,
            )
        )

        # 一時配列へ再投影
        temp_dst = np.full(
            (out_height, out_width),
            config.NODATA_VALUE,
            dtype=np.float32,
        )

        reproject(
            source=src_array,
            destination=temp_dst,
            src_transform=src_transform,
            src_crs=config.SRC_CRS,
            src_nodata=config.NODATA_VALUE,
            dst_transform=dst_transform,
            dst_crs=config.OUTPUT_CRS,
            dst_nodata=config.NODATA_VALUE,
            resampling=Resampling.nearest,
        )

        # 有効値だけ上書き
        valid_mask = temp_dst != config.NODATA_VALUE
        out_array[valid_mask] = temp_dst[valid_mask]

    # -----------------------------------------------------
    # GeoTIFF 保存
    # -----------------------------------------------------
    profile = build_gtiff_profile(
        width=out_width,
        height=out_height,
        dtype=config.OUTPUT_DTYPE,
        crs=config.OUTPUT_CRS,
        transform=dst_transform,
        nodata_value=config.NODATA_VALUE,
        compress=config.COMPRESS,
        tiled=config.GTIFF_TILED,
        blockxsize=config.GTIFF_BLOCKXSIZE,
        blockysize=config.GTIFF_BLOCKYSIZE,
    )

    with rasterio.open(config.OUTPUT_TIFF_PATH, "w", **profile) as dst:
        dst.write(out_array.astype(np.float32), 1)
        # ArcGISに格子点データ（Point）であることを明示
        dst.update_tags(AREA_OR_POINT="Point")

    log_info(f"GeoTIFF 保存完了: {config.OUTPUT_TIFF_PATH}")

    # -----------------------------------------------------
    # タイル矩形 GeoParquet 保存
    # -----------------------------------------------------
    if config.EXPORT_TILE_EXTENT_PARQUET:
        save_tile_extent_parquet(
            records=tile_extent_records,
            out_path=config.OUTPUT_TILE_EXTENT_PATH,
            crs=config.OUTPUT_CRS,
        )


if __name__ == "__main__":
    main()