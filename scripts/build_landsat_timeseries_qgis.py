#!/usr/bin/env python3
"""Build an observed Landsat March–October NDVI project with QGIS Python."""
from __future__ import annotations

import argparse
import calendar
import json
from pathlib import Path

from qgis.PyQt.QtGui import QColor
from qgis.core import (
    Qgis, QgsApplication, QgsDateTimeRange, QgsLayoutExporter, QgsLayoutItemPicture,
    QgsLayoutPoint, QgsLayoutSize, QgsPalettedRasterRenderer, QgsProject, QgsReferencedRectangle,
)

from build_ndvi_timeseries_qgis import add_raster, color_legend, date_range, label, layout_page, map_panel
from build_spring_review_qgis import VIRIDIS, ndvi_renderer


def build(directory):
    report = json.loads((directory / "review.json").read_text(encoding="utf-8"))
    year = report["year"]
    project = QgsProject.instance()
    project.clear()
    project.setTitle(f"Clayton study area | Observed Landsat, March–October {year}")
    project.setFilePathStorage(Qgis.FilePathType.Relative)
    months = project.layerTreeRoot().addGroup("Observed monthly NDVI | select one month")
    monthly_layers = []
    for row in report["months"]:
        prefix = row["month"]
        layer = add_raster(project, months, directory / f"monthly/Landsat_observed_mean_{prefix}.tif",
                           f"{prefix} | Mean observed NDVI")
        layer.setRenderer(ndvi_renderer(layer, report["ndvi_display_limits"]))
        monthly_layers.append(layer)
    months.setIsMutuallyExclusive(True, 4)
    extent = monthly_layers[0].extent()
    project.setCrs(monthly_layers[0].crs())
    project.viewSettings().setDefaultViewExtent(QgsReferencedRectangle(extent, project.crs()))
    monthly_layers[0].saveNamedStyle(str(directory / "NDVI_shared.qml"))

    acquisitions = project.layerTreeRoot().addGroup("Acquisition NDVI | select an actual Landsat date")
    for day in report["acquisition_dates"]:
        layer = add_raster(project, acquisitions, directory / f"acquisitions/Landsat_NDVI_{day}.tif", day)
        layer.setRenderer(ndvi_renderer(layer, report["ndvi_display_limits"]))
    acquisitions.setIsMutuallyExclusive(True, report["acquisition_dates"].index(report["curve_dates"][0]))
    acquisitions.setItemVisibilityChecked(False)
    acquisitions.setExpanded(False)

    temporal_group = project.layerTreeRoot().addGroup("Calendar observations | use Temporal Controller")
    temporal_layer = add_raster(project, temporal_group,
                                directory / f"annual/Landsat_observed_NDVI_{year}_Mar_Oct.tif",
                                "Observed NDVI | Blank on dates without clear Landsat")
    temporal_layer.setRenderer(ndvi_renderer(temporal_layer, report["ndvi_display_limits"]))
    ranges = {index: date_range(day) for index, day in enumerate(report["calendar_dates"], 1)}
    if temporal_layer.bandCount() != len(ranges):
        raise ValueError("The Landsat calendar stack has the wrong number of bands")
    temporal = temporal_layer.temporalProperties()
    temporal.setMode(Qgis.RasterTemporalMode.FixedRangePerBand)
    temporal.setFixedRangePerBand(ranges)
    temporal.setIsActive(True)
    temporal_group.setItemVisibilityChecked(False)
    project.timeSettings().setTemporalRange(QgsDateTimeRange(
        ranges[1].begin(), ranges[len(ranges)].end(), True, False))
    project.timeSettings().setTimeStep(1)
    project.timeSettings().setTimeStepUnit(Qgis.TemporalUnit.Days)
    project.timeSettings().setFramesPerSecond(3)

    counts = project.layerTreeRoot().addGroup("Monthly clear-observation counts | select one month")
    for row in report["months"]:
        prefix = row["month"]
        layer = add_raster(project, counts, directory / f"monthly/Landsat_clear_count_{prefix}.tif",
                           f"{prefix} | Clear observations (0 to {row['acquisition_dates']})")
        layer.setRenderer(ndvi_renderer(layer, (0, max(1, row["acquisition_dates"]))))
    counts.setIsMutuallyExclusive(True, 4)
    counts.setItemVisibilityChecked(False)
    counts.setExpanded(False)
    footprint = project.layerTreeRoot().addGroup("Curve footprint | same pixels on all plotted dates")
    layer = add_raster(project, footprint, directory / "curve_common_pixels.tif", "Fixed curve footprint")
    layer.setRenderer(QgsPalettedRasterRenderer(layer.dataProvider(), 1, [
        QgsPalettedRasterRenderer.Class(0, QColor("#e0e4e9"), "Outside curve footprint"),
        QgsPalettedRasterRenderer.Class(1, QColor("#315b9a"), "Inside curve footprint"),
    ]))
    footprint.setItemVisibilityChecked(False)
    footprint.setExpanded(False)

    overview = layout_page(project, "Observed monthly NDVI", 400, 222)
    label(overview, f"Clayton study area | Observed Landsat NDVI, {year}", 8, 5, 384, 19,
          bold=True, height=12)
    label(overview, "Mean of QA-clear Landsat observations | March through October | One NDVI scale", 8, 20, 384, 11)
    for index, (layer, row) in enumerate(zip(monthly_layers, report["months"])):
        month = index + 3
        x = 8 + 98 * (index % 4)
        y = 33 + 78 * (index // 4)
        label(overview, calendar.month_name[month], x, y, 92, 12, bold=True, center=True)
        map_panel(overview, project, layer, extent, x + 15, y + 9, 62)
        label(overview, f"Supported area {row['supported_area_fraction']:.1%}", x, y + 71, 92, 9,
              height=6, center=True)
    color_legend(overview, report["ndvi_display_limits"], VIRIDIS, 197,
                 note=f"5.01 km square at 30 m | At least {report['monthly_minimum_clear_observations']} clear observations per pixel per month | Gray indicates NoData")
    curve = layout_page(project, "Observed NDVI curve", 320, 210)
    picture = QgsLayoutItemPicture(curve)
    curve.addLayoutItem(picture)
    picture.attemptMove(QgsLayoutPoint(3, 3))
    picture.attemptResize(QgsLayoutSize(314, 199))
    picture.setPicturePath(str(directory / "landsat_observed_curve.png"))
    label(curve, "Points use a fixed set of locations | Monthly maps use clear observations available at each pixel", 6, 202, 308, 8)
    target = directory / "landsat_observed_time_series.qgz"
    if not project.write(str(target)):
        raise RuntimeError(f"Could not save {target}")
    settings = QgsLayoutExporter.ImageExportSettings()
    settings.dpi = 160
    if QgsLayoutExporter(overview).exportToImage(str(directory / "landsat_observed_monthly_maps.png"), settings) != QgsLayoutExporter.Success:
        raise RuntimeError("Could not render the observed monthly maps")
    expected = len(project.mapLayers())
    project.clear()
    if (not project.read(str(target)) or len(project.mapLayers()) != expected
            or len(project.layoutManager().layouts()) != 2
            or not all(layer.isValid() for layer in project.mapLayers().values())):
        raise RuntimeError("The observed Landsat project could not be reopened completely")
    for layer in project.mapLayers().values():
        properties = layer.temporalProperties()
        if properties.isActive():
            for band, interval in ranges.items():
                if properties.bandForTemporalRange(layer, interval) != band:
                    raise RuntimeError(f"Calendar mismatch for band {band}")
    print(f"{target} ({expected} layers, 2 layouts, {len(ranges)} calendar dates)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    app = QgsApplication([], True)
    app.initQgis()
    build(args.directory.resolve())
