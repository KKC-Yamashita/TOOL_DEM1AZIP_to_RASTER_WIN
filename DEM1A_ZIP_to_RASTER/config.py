# -*- coding: utf-8 -*-
"""
config.py
ユーザー設定
"""

from __future__ import annotations

from pathlib import Path


# =========================================================
# 入出力設定
# =========================================================
# DEM1A の ZIP 格納フォルダ
TARGET_ZIP_DIR = Path(r"D:\ArcGIS_Project\2604_Kumamoto_Hikawa\03_GSI_DEM1A\01_ORG")

# 出力フォルダ
OUTPUT_DIR = Path(r"D:\ArcGIS_Project\2604_Kumamoto_Hikawa\03_GSI_DEM1A\02_DEM1A_TIFF")

# 出力 GeoTIFF 名
OUTPUT_TIFF_NAME = "Kumamoto_Hikawa_DEM1A.tif"

# XML ごとの想定タイル矩形 GeoParquet 名
OUTPUT_TILE_EXTENT_NAME = "Kumamoto_Hikawa_DEM1A_tile_extent.parquet"


# =========================================================
# 抽出範囲設定
# 経緯度（分指定は10進数表記）
# ※ 分指定を残したいので式のまま記述
# =========================================================
WEST = 130 + 36 / 60.0
EAST = 130 + 44 / 60.0
SOUTH = 32 + 31 / 60.0
NORTH = 32 + 38 / 60.0


# =========================================================
# CRS 設定
# =========================================================
# NOTE:
# 本データ（DEM1A XML）は gml:srsName="jgd2024.bl" を持つが、
# 現時点では JGD2024 の正式な EPSG コードが未整備のため、
# 暫定的に EPSG:6668（JGD2011 地理座標系）を使用する。
#
# JGD2011 と JGD2024 の差異は本対象範囲では数cmレベルと小さいため、
# 本処理（可視化・解析）への影響は実用上無視可能と判断する。
#
# 将来的に JGD2024 の正式な EPSG が利用可能になった場合は、
# SRC_CRS / OUTPUT_CRS を更新すること。
SRC_CRS = "EPSG:6668"
OUTPUT_CRS = "EPSG:6668"


# =========================================================
# 出力設定
# =========================================================
NODATA_VALUE = -9999.0
OUTPUT_DTYPE = "float32"
COMPRESS = "LZW"

# GeoTIFF タイル設定
GTIFF_TILED = True
GTIFF_BLOCKXSIZE = 256
GTIFF_BLOCKYSIZE = 256


# =========================================================
# ログ・追加出力設定
# =========================================================
VERBOSE = True

# 将来用:
# 確認用ポイント GeoParquet などを追加したくなった時に使う想定
EXPORT_TILE_EXTENT_PARQUET = True


# =========================================================
# 内部利用
# =========================================================
TARGET_BBOX = (WEST, SOUTH, EAST, NORTH)
OUTPUT_TIFF_PATH = OUTPUT_DIR / OUTPUT_TIFF_NAME
OUTPUT_TILE_EXTENT_PATH = OUTPUT_DIR / OUTPUT_TILE_EXTENT_NAME