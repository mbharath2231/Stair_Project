#!/usr/bin/env python3
"""Build a portable one-year Clayton NDVI project with temporal band navigation."""
from __future__ import annotations

import argparse
import calendar
from datetime import date, timedelta
import json
from pathlib import Path

from qgis.PyQt.QtCore import QDateTime, Qt
from qgis.PyQt.QtGui import QColor, QFont
from qgis.core import (
    Qgis, QgsApplication, QgsDateTimeRange, QgsFillSymbol, QgsLayoutExporter, QgsLayoutItemLabel,
    QgsLayoutItemMap, QgsLayoutItemPicture, QgsLayoutItemShape, QgsLayoutPoint,
    QgsLayoutSize, QgsPalettedRasterRenderer, QgsPrintLayout, QgsProject,
    QgsRasterLayer, QgsReferencedRectangle,
)

from build_spring_review_qgis import VIRIDIS, ndvi_renderer
from build_watershed_qgis import diverging

SOURCE_CLASSES = (
    (1, "#427baa", "Observed Landsat"), (2, "#187963", "Two-pair fusion"),
    (3, "#f5b547", "Single-pair fusion"), (4, "#8e639c", "Landsat interpolation"),
)


def date_range(day):
    start = date.fromisoformat(day)
    return QgsDateTimeRange(
        QDateTime.fromString(start.isoformat() + "T00:00:00Z", Qt.ISODate),
        QDateTime.fromString((start + timedelta(days=1)).isoformat() + "T00:00:00Z", Qt.ISODate),
        True, False,
    )


def label(layout, value, x, y, width, size=10, *, bold=False, height=8, center=False):
    item = QgsLayoutItemLabel(layout)
    item.setText(value)
    item.setFont(QFont("Arial", size, QFont.Bold if bold else QFont.Normal))
    item.setFontColor(QColor("#1d2b36"))
    item.setHAlign(Qt.AlignHCenter if center else Qt.AlignLeft)
    layout.addLayoutItem(item)
    item.attemptMove(QgsLayoutPoint(x, y))
    item.attemptResize(QgsLayoutSize(width, height))
    return item


def layout_page(project, name, width, height):
    layout = QgsPrintLayout(project)
    layout.initializeDefaults()
    layout.setName(name)
    layout.pageCollection().pages()[0].setPageSize(QgsLayoutSize(width, height))
    project.layoutManager().addLayout(layout)
    return layout


def map_panel(layout, project, layer, extent, x, y, side):
    panel = QgsLayoutItemMap(layout)
    layout.addLayoutItem(panel)
    panel.attemptMove(QgsLayoutPoint(x, y))
    panel.attemptResize(QgsLayoutSize(side, side))
    panel.setCrs(project.crs())
    panel.setLayers([layer])
    panel.setKeepLayerSet(True)
    panel.setBackgroundColor(QColor("#e0e4e9"))
    panel.setFrameEnabled(True)
    panel.zoomToExtent(extent)
    return panel


def color_legend(layout, bounds, colors, y, *, title="NDVI", note="Gray indicates NoData"):
    label(layout, title, 10, y, 28, bold=True)
    low, high = bounds
    for index, color in enumerate(colors):
        x = 40 + 43 * index
        swatch = QgsLayoutItemShape(layout)
        swatch.setShapeType(QgsLayoutItemShape.Rectangle)
        swatch.setSymbol(QgsFillSymbol.createSimple({"color": color, "outline_style": "no"}))
        swatch.setFrameEnabled(False)
        layout.addLayoutItem(swatch)
        swatch.attemptMove(QgsLayoutPoint(x, y + 2))
        swatch.attemptResize(QgsLayoutSize(10, 4))
        value = low + (high - low) * index / (len(colors) - 1)
        text = "0.00" if abs(value) < 1e-9 else f"{value:.2f}"
        label(layout, text, x + 12, y, 27, size=9)
    label(layout, note, 10, y + 9, 380, size=9)


def add_raster(project, group, path, name):
    layer = QgsRasterLayer(str(path), name)
    if not layer.isValid():
        raise ValueError(f"Cannot open {path}")
    project.addMapLayer(layer, False)
    node = group.addLayer(layer)
    node.setExpanded(False)
    return layer


def build(directory):
    report = json.loads((directory / "review.json").read_text(encoding="utf-8"))
    year = report["year"]
    project = QgsProject.instance()
    project.clear()
    project.setTitle(f"Clayton study area | NDVI time series {year}")
    project.setFilePathStorage(Qgis.FilePathType.Relative)
    months = project.layerTreeRoot().addGroup("Monthly mean NDVI | select one month")
    monthly_layers = []
    for month in range(1, 13):
        layer = add_raster(project, months, directory / f"monthly/NDVI_mean_{year}-{month:02d}.tif",
                           f"{year}-{month:02d} | {calendar.month_name[month]} mean NDVI")
        layer.setRenderer(ndvi_renderer(layer, report["ndvi_display_limits"]))
        monthly_layers.append(layer)
    months.setIsMutuallyExclusive(True, 6)
    months.setExpanded(True)
    extent = monthly_layers[0].extent()
    project.setCrs(monthly_layers[0].crs())
    project.viewSettings().setDefaultViewExtent(QgsReferencedRectangle(extent, project.crs()))
    monthly_layers[0].saveNamedStyle(str(directory / "NDVI_shared.qml"))

    temporal_group = project.layerTreeRoot().addGroup("Daily time series | use Temporal Controller")
    bands = {index: date_range(day) for index, day in enumerate(report["dates"], 1)}
    annual_names = (
        (f"NDVI_{year}.tif", "NDVI product"),
        (f"Landsat_NDVI_{year}.tif", "QA-clear Landsat NDVI"),
        (f"MODIS_NDVI_{year}.tif", "QA-screened MODIS NDVI on 30 m grid"),
        (f"NDVI_source_{year}.tif", "NDVI source flags"),
    )
    temporal_layers = []
    for filename, name in annual_names:
        layer = add_raster(project, temporal_group, directory / "annual" / filename, name)
        if layer.bandCount() != len(bands):
            raise ValueError(f"Date count mismatch in {filename}")
        if name == "NDVI source flags":
            classes = [QgsPalettedRasterRenderer.Class(value, QColor(color), text)
                       for value, color, text in SOURCE_CLASSES]
            layer.setRenderer(QgsPalettedRasterRenderer(layer.dataProvider(), 1, classes))
        else:
            layer.setRenderer(ndvi_renderer(layer, report["ndvi_display_limits"]))
        temporal = layer.temporalProperties()
        temporal.setMode(Qgis.RasterTemporalMode.FixedRangePerBand)
        temporal.setFixedRangePerBand(bands)
        temporal.setIsActive(True)
        temporal_layers.append(layer)
    temporal_group.setIsMutuallyExclusive(True, 0)
    temporal_group.setItemVisibilityChecked(False)
    temporal_group.setExpanded(True)
    project.timeSettings().setTemporalRange(QgsDateTimeRange(
        bands[1].begin(), bands[len(bands)].end(), True, False))
    project.timeSettings().setTimeStep(1)
    project.timeSettings().setTimeStepUnit(Qgis.TemporalUnit.Days)
    project.timeSettings().setFramesPerSecond(5)

    coverage = project.layerTreeRoot().addGroup("Monthly supported days | select one month")
    for month, row in enumerate(report["months"], 1):
        layer = add_raster(project, coverage, directory / f"monthly/NDVI_valid_days_{year}-{month:02d}.tif",
                           f"{year}-{month:02d} | Supported days (0 to {row['calendar_days']})")
        layer.setRenderer(ndvi_renderer(layer, (0, row["calendar_days"])))
    coverage.setIsMutuallyExclusive(True, 6)
    coverage.setItemVisibilityChecked(False)
    coverage.setExpanded(False)

    changes = project.layerTreeRoot().addGroup("Seasonal NDVI changes | select one comparison")
    change_layers = []
    for row in report["seasonal_changes"]:
        layer = add_raster(project, changes, directory / row["raster"],
                           f"{row['later_month']} minus {row['earlier_month']} | Monthly mean NDVI")
        layer.setRenderer(diverging(layer, report["change_display_limits"][1]))
        change_layers.append(layer)
    changes.setIsMutuallyExclusive(True, 0)
    changes.setItemVisibilityChecked(False)

    overview = layout_page(project, "Monthly NDVI", 400, 300)
    label(overview, f"Clayton study area | NDVI through {year}", 8, 5, 384, 20, bold=True, height=12)
    label(overview, "Monthly mean of supported daily NDVI | One color scale for all months", 8, 20, 384, 11)
    for index, (layer, row) in enumerate(zip(monthly_layers, report["months"])):
        x = 8 + 98 * (index % 4)
        y = 33 + 78 * (index // 4)
        label(overview, calendar.month_name[index + 1], x, y, 92, 12, bold=True, center=True)
        map_panel(overview, project, layer, extent, x + 15, y + 9, 62)
        label(overview, f"Supported area {row['valid_area_fraction']:.0%}", x, y + 71, 92, 9,
              height=6, center=True)
    # Each column has identical spatial extent. The legend sits beneath the third row.
    color_legend(overview, report["ndvi_display_limits"], VIRIDIS, 279,
                 note=f"30 m grid, 5.01 km square | Each pixel needs estimates on at least {report['monthly_minimum_supported_day_fraction']:.0%} of the month's days")

    seasonal = layout_page(project, "Seasonal NDVI changes", 400, 225)
    label(seasonal, f"Clayton study area | Seasonal NDVI change, {year}", 8, 5, 384, 19, bold=True, height=12)
    label(seasonal, "Monthly means on shared valid pixels | Positive values indicate increased NDVI", 8, 20, 384, 11)
    for index, (layer, row) in enumerate(zip(change_layers, report["seasonal_changes"])):
        x = 15 + 195 * index
        label(seasonal, f"{row['later_month']} minus {row['earlier_month']}", x, 32, 175, 13, bold=True, center=True)
        map_panel(seasonal, project, layer, extent, x + 28, 45, 120)
        label(seasonal, f"Shared supported area {row['shared_valid_area_fraction']:.0%}", x, 169, 175, 10, center=True)
    color_legend(seasonal, report["change_display_limits"], ("#244da3", "#f8f8f8", "#b64425"), 190,
                 title="NDVI change", note="Blue indicates decreased NDVI and orange indicates increased NDVI. Gray indicates NoData.")

    curve = layout_page(project, "NDVI curve", 320, 210)
    picture = QgsLayoutItemPicture(curve)
    curve.addLayoutItem(picture)
    picture.attemptMove(QgsLayoutPoint(3, 3))
    picture.attemptResize(QgsLayoutSize(314, 199))
    picture.setPicturePath(str(directory / "ndvi_time_series.png"))
    label(curve, "Landsat and MODIS residual fusion | Source flags and coverage travel with the NDVI values", 6, 202, 308, 8)

    target = directory / "clayton_ndvi_time_series.qgz"
    if not project.write(str(target)):
        raise RuntimeError(f"Could not save {target}")
    settings = QgsLayoutExporter.ImageExportSettings()
    settings.dpi = 160
    for layout, filename in ((overview, "monthly_ndvi_maps.png"), (seasonal, "seasonal_change.png")):
        if QgsLayoutExporter(layout).exportToImage(str(directory / filename), settings) != QgsLayoutExporter.Success:
            raise RuntimeError(f"Could not render {filename}")
    expected_layers = len(project.mapLayers())
    project.clear()
    if (not project.read(str(target)) or len(project.mapLayers()) != expected_layers
            or len(project.layoutManager().layouts()) != 3
            or not all(layer.isValid() for layer in project.mapLayers().values())):
        raise RuntimeError("The saved project did not reopen completely")
    for layer in project.mapLayers().values():
        temporal = layer.temporalProperties()
        if temporal.isActive():
            if len(temporal.fixedRangePerBand()) != len(bands):
                raise RuntimeError(f"Temporal calendar lost for {layer.name()}")
            for band, interval in bands.items():
                if temporal.bandForTemporalRange(layer, interval) != band:
                    raise RuntimeError(f"Temporal band mismatch for {layer.name()}, {band}")
    print(f"{target} ({expected_layers} layers, 3 layouts, {len(bands)} temporal dates)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    app = QgsApplication([], True)
    app.initQgis()
    build(args.directory.resolve())
