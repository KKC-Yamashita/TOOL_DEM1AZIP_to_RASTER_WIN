# -*- coding: utf-8 -*-
"""
dem_utils.py
DEM1A XML / ZIP 読取・tupleList 復元・補助関数群
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

import geopandas as gpd
import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import box


# =========================================================
# dataclass
# =========================================================
@dataclass
class TileCandidate:
    """Envelope 先読みで抽出した候補タイル情報"""
    zip_path: Path
    xml_name: str
    west: float
    south: float
    east: float
    north: float


@dataclass
class TileGridSpec:
    """1タイル分の格子仕様"""
    nx: int
    ny: int
    dx: float
    dy: float


# =========================================================
# 基本ユーティリティ
# =========================================================
def log_info(message: str) -> None:
    print(f"[INFO] {message}")


def log_warn(message: str) -> None:
    print(f"[WARN] {message}")


def local_name(tag: str) -> str:
    """名前空間付きタグからローカル名だけ返す"""
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def bbox_intersects(
    west1: float,
    south1: float,
    east1: float,
    north1: float,
    west2: float,
    south2: float,
    east2: float,
    north2: float,
) -> bool:
    """2つの bbox が交差するか判定"""
    return not (
        east1 <= west2 or east2 <= west1 or
        north1 <= south2 or north2 <= south1
    )


def find_first_text(root: ET.Element, tag_name: str) -> str | None:
    """ローカル名一致の最初の要素 text を返す"""
    for elem in root.iter():
        if local_name(elem.tag) == tag_name:
            return (elem.text or "").strip()
    return None


def find_first_elem(root: ET.Element, tag_name: str) -> ET.Element | None:
    """ローカル名一致の最初の要素を返す"""
    for elem in root.iter():
        if local_name(elem.tag) == tag_name:
            return elem
    return None


# =========================================================
# XML 読取
# =========================================================
def read_xml_root_from_zip(zip_path: Path, xml_name: str) -> ET.Element:
    """ZIP 内 XML を読み込んで root を返す"""
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(xml_name) as f:
            data = f.read()
    return ET.parse(BytesIO(data)).getroot()


def find_xml_names_in_zip(zip_path: Path) -> list[str]:
    """ZIP 内の XML 名一覧を返す"""
    with zipfile.ZipFile(zip_path, "r") as zf:
        return [name for name in zf.namelist() if name.lower().endswith(".xml")]


# =========================================================
# XML 解析
# =========================================================
def parse_envelope(root: ET.Element) -> tuple[float, float, float, float]:
    """
    lowerCorner / upperCorner を読む
    XML 上は通常 'lat lon' の並び

    Returns
    -------
    (west, south, east, north)
    """
    lower = find_first_text(root, "lowerCorner")
    upper = find_first_text(root, "upperCorner")

    if lower is None or upper is None:
        raise ValueError("lowerCorner / upperCorner が見つかりません。")

    south, west = map(float, lower.split())
    north, east = map(float, upper.split())
    return west, south, east, north


def parse_grid_size(root: ET.Element) -> tuple[int, int]:
    """
    gml:high から格子数を読む
    例: high = '1124 749' -> nx=1125, ny=750
    """
    high_text = find_first_text(root, "high")
    if high_text is None:
        raise ValueError("gml:high が見つかりません。")

    high_x, high_y = map(int, high_text.split())
    return high_x + 1, high_y + 1


def parse_sequence_rule_order(root: ET.Element) -> str:
    """
    gml:sequenceRule の order 属性を取得
    通常は '+x-y'
    """
    for elem in root.iter():
        if local_name(elem.tag) == "sequenceRule":
            return elem.attrib.get("order", "").strip()
    return ""


def parse_start_point(root: ET.Element) -> tuple[int, int]:
    """
    gml:startPoint を読む
    なければ (0, 0) を返す
    """
    text = find_first_text(root, "startPoint")
    if not text:
        return 0, 0

    parts = text.replace(",", " ").split()
    if len(parts) < 2:
        return 0, 0

    return int(parts[0]), int(parts[1])


def parse_tuplelist_values(root: ET.Element) -> list[float]:
    """
    gml:tupleList から標高値だけを一次元 list で取得
    各行は通常 '種別,値' の想定
    """
    tuple_elem = find_first_elem(root, "tupleList")
    if tuple_elem is None or tuple_elem.text is None:
        raise ValueError("gml:tupleList が見つかりません。")

    values: list[float] = []

    for line in tuple_elem.text.splitlines():
        line = line.strip()
        if not line:
            continue

        try:
            z_value = float(line.rsplit(",", 1)[1].strip())
        except Exception:
            continue

        values.append(z_value)

    return values


def parse_tuplelist_to_array(
    root: ET.Element,
    nx: int,
    ny: int,
    nodata_value: float,
) -> np.ndarray:
    """
    tupleList を 2次元配列へ復元する

    前提仕様
    --------
    - gml:startPoint = (sx, sy) は tupleList 先頭値が入る格子位置
    - sequenceRule order = '+x-y'
      => 左→右、上→下
    - 配列末尾まで行ったら先頭へ循環

    注意
    ----
    lowerCorner / upperCorner は配置用であり、
    データの並び順自体は startPoint + tupleList で決まる
    """
    order = parse_sequence_rule_order(root)
    if order and order != "+x-y":
        raise ValueError(f"未対応の sequenceRule order です: {order}")

    start_x, start_y = parse_start_point(root)
    values = parse_tuplelist_values(root)

    total_cells = nx * ny
    start_linear_index = start_y * nx + start_x

    array_2d = np.full((ny, nx), nodata_value, dtype=np.float32)

    for idx, value in enumerate(values):
        linear_index = (start_linear_index + idx) % total_cells
        row = linear_index // nx
        col = linear_index % nx
        array_2d[row, col] = value

    return array_2d


# =========================================================
# タイル探索
# =========================================================
def read_envelope_from_xml_in_zip_fast(
    zip_path: Path,
    xml_name: str,
) -> tuple[float, float, float, float] | None:
    """
    XML 全体を構築せず、Envelope だけ先読みする
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open(xml_name) as f:
                lower_text = None
                upper_text = None

                for _, elem in ET.iterparse(f, events=("end",)):
                    tag = local_name(elem.tag)
                    text = (elem.text or "").strip()

                    if tag == "lowerCorner" and lower_text is None:
                        lower_text = text
                    elif tag == "upperCorner" and upper_text is None:
                        upper_text = text
                        break

                    elem.clear()

        if lower_text is None or upper_text is None:
            return None

        south, west = map(float, lower_text.split())
        north, east = map(float, upper_text.split())
        return west, south, east, north

    except Exception:
        return None


def collect_intersecting_tiles(
    zip_files: list[Path],
    target_bbox: tuple[float, float, float, float],
) -> list[TileCandidate]:
    """
    ZIP 群を走査し、Envelope が target_bbox に交差する XML を返す
    """
    target_west, target_south, target_east, target_north = target_bbox
    hits: list[TileCandidate] = []

    for zip_path in zip_files:
        xml_names = find_xml_names_in_zip(zip_path)

        for xml_name in xml_names:
            envelope = read_envelope_from_xml_in_zip_fast(zip_path, xml_name)
            if envelope is None:
                log_warn(f"Envelope読取失敗: {zip_path.name} / {xml_name}")
                continue

            west, south, east, north = envelope

            if bbox_intersects(
                west, south, east, north,
                target_west, target_south, target_east, target_north,
            ):
                hits.append(
                    TileCandidate(
                        zip_path=zip_path,
                        xml_name=xml_name,
                        west=west,
                        south=south,
                        east=east,
                        north=north,
                    )
                )

    return hits


# =========================================================
# 格子仕様・transform
# =========================================================
def build_tile_grid_spec_from_root(root: ET.Element) -> TileGridSpec:
    """1タイルの格子仕様を root から作る"""
    nx, ny = parse_grid_size(root)
    west, south, east, north = parse_envelope(root)

    dx = (east - west) / (nx - 1)
    dy = (north - south) / (ny - 1)

    return TileGridSpec(nx=nx, ny=ny, dx=dx, dy=dy)


def build_src_transform(
    west: float,
    south: float,
    east: float,
    north: float,
    nx: int,
    ny: int,
):
    """
    タイル Envelope から source transform を作る
    格子点が画素中心に来るよう half-pixel シフト
    """
    dx = (east - west) / (nx - 1)
    dy = (north - south) / (ny - 1)

    return from_origin(
        west - dx / 2.0,
        north + dy / 2.0,
        dx,
        dy,
    )


def build_dst_transform(
    west: float,
    south: float,
    east: float,
    north: float,
    dx: float,
    dy: float,
):
    """
    出力範囲と格子間隔から destination transform を作る
    """
    return from_origin(
        west - dx / 2.0,
        north + dy / 2.0,
        dx,
        dy,
    )


def calc_output_grid_size(
    west: float,
    south: float,
    east: float,
    north: float,
    dx: float,
    dy: float,
) -> tuple[int, int]:
    """出力 raster の width / height を返す"""
    out_nx = int(round((east - west) / dx)) + 1
    out_ny = int(round((north - south) / dy)) + 1
    return out_nx, out_ny


def build_gtiff_profile(
    width: int,
    height: int,
    dtype: str,
    crs: str,
    transform,
    nodata_value: float,
    compress: str,
    tiled: bool,
    blockxsize: int,
    blockysize: int,
) -> dict:
    """GeoTIFF profile を作る"""
    return {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata_value,
        "compress": compress,
        "tiled": tiled,
        "blockxsize": blockxsize,
        "blockysize": blockysize,
    }


# =========================================================
# 出力補助
# =========================================================
def build_tile_extent_record(
    zip_name: str,
    xml_name: str,
    west: float,
    south: float,
    east: float,
    north: float,
    start_x: int,
    start_y: int,
    order: str,
    nx: int,
    ny: int,
) -> dict:
    """
    タイル矩形 GeoParquet 用の1レコードを作る
    将来、確認用属性を追加しやすいよう関数化
    """
    return {
        "zip_name": zip_name,
        "xml_name": Path(xml_name).name,
        "raw_west": west,
        "raw_south": south,
        "raw_east": east,
        "raw_north": north,
        "start_x": start_x,
        "start_y": start_y,
        "order": order,
        "nx": nx,
        "ny": ny,
        "geometry": box(west, south, east, north),
    }


def save_tile_extent_parquet(
    records: list[dict],
    out_path: Path,
    crs: str,
) -> None:
    """タイル矩形 GeoParquet を保存"""
    if not records:
        log_warn("タイル矩形出力対象がありません。")
        return

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
    gdf.to_parquet(out_path, index=False)
    log_info(f"タイル矩形 GeoParquet 保存完了: {out_path}")